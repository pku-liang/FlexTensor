import tvm
import torch
torch.set_printoptions(threshold=100000)
import torch_geometric

import model

from auto_schedule.scheduler import schedule_with_config_ops
from auto_schedule.utils import any_factor_split


def gemm(A, B, transposeA=False, transposeB=False):
    """Matrix multiplies matrix

    Args:
    -----------------------------
    A: tvm.tensor.Tensor
        shape [height, width]
    B: tvm.tensor.Tensor
        shape [width, length]
    transposeA: (optional:False) bool
    transposeB: (optional:False) bool
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [height, length]
    -----------------------------
    """
    if transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert (A.shape[0].value == B.shape[1].value)
        return tvm.compute((A.shape[1], B.shape[0]), lambda i, j: tvm.sum(A[k, i] * B[j, k], axis=k))
    elif transposeA and not transposeB:
        k = tvm.reduce_axis((0, B.shape[0]))
        assert (A.shape[0].value == B.shape[0].value)
        return tvm.compute((A.shape[1], B.shape[1]), lambda i, j: tvm.sum(A[k, i] * B[k, j], axis=k))
    elif not transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert (A.shape[1].value == B.shape[1].value)
        return tvm.compute((A.shape[0], B.shape[0]), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k))
    else:
        k = tvm.reduce_axis((0, B.shape[0]))
        assert (A.shape[1].value == B.shape[0].value)
        return tvm.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k))


def zero_pad2d(inputs, padding=0):
    """Zero padding for 2d tensor

    Args:
    -----------------------------
    inputs : tvm.tensor.Tensor
        shape [batch, channel, height, width]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, channel, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    assert isinstance(padding, tuple)
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert (len(padding) == 4)

    padding_zero = 0.0 if "float" in inputs.dtype else 0

    batch_size, in_channel, height, width = inputs.shape
    return tvm.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.if_then_else(
                            tvm.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                            inputs[b, c, h - padding[0], w - padding[2]],
                            padding_zero
                            ),
        name='Padding'
        )


def conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : tvm.tensor.Tensor
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) tvm.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert ((channel_per_group * groups).value == in_channel.value)
    out_channel_per_group = out_channel // groups
    assert ((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.expr.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.expr.IntImm)) else dilation
    assert (isinstance(stride, tuple) and len(stride) == 2)
    assert (isinstance(padding, tuple) and len(padding) == 2)
    assert (isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rc = tvm.reduce_axis((0, channel_per_group), name="rc")
    rh = tvm.reduce_axis((0, k_h), name="rh")
    rw = tvm.reduce_axis((0, k_w), name="rw")

    padded = zero_pad2d(inputs, padding=padding)
    output = tvm.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda b, c, h, w: tvm.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
            * weight[c, rc, rh, rw]),
            axis=[rc, rw, rh]
        )
    )
    if bias is not None:
        output = tvm.compute(
            (batch_size, out_channel, out_h, out_w),
            lambda b, c, h, w: output[b, c, h, w] + bias[c]
        )
    return output


def schedule_ops(ops, bufs, config):
    s, bufs = schedule_with_config_ops(ops, bufs, config)
    return s, bufs

"""[
    [
        {
            "spatial": [[32, 2, 16, 2], [8, 4, 16, 4]], 
            "reduce": [[2048, 1, 1]], 
            "inline": [], 
            "unroll": [[1, 0]]
        }
    ], 
    {
        "spatial": [], 
        "reduce": [], 
        "inline": [[0]], 
        "unroll": []
    }
   ]
"""
def gemm_config(M, N, K, logits):
    sy = any_factor_split(M, 4)
    sx = any_factor_split(N, 4)
    sk = any_factor_split(K, 4)
    unroll = []
    for i in range(1):
        for j in range(11):
            unroll.append([i, 2**j])
    


def graph_gemm(M, N, K):
    x = torch.FloatTensor(
        [
            [M, 0],  # 0, tensor nodes
            [K, 0],  # 1
            [1, 0],  # 2
            [K, 0],  # 3
            [N, 0],  # 4
            [1, 0],  # 5
            [M, 0],  # 6
            [N, 0],  # 7
            [1, 0],  # 8
            [0, M],  # 9, index nodes
            [0, N],  # 10
            [0, K]   # 11
        ]
    )
    edge_index = torch.LongTensor(
        [
            [0, 1],  # stride edge
            [1, 0],
            [1, 2],
            [2, 1],
            [3, 4],
            [4, 3],
            [4, 5],
            [5, 4],
            [6, 7],
            [7, 6],
            [7, 8],
            [8, 7],
            [0, 9],  # read edge
            [9, 0],
            [1, 10],
            [10, 1],
            [3, 10],
            [10, 3],
            [4, 11],
            [11, 4],
            [6, 9],  # write edge
            [9, 6],
            [7, 11],
            [11, 7],
            [0, 0],  # self edge
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],
            [8, 8],
            [9, 9],
            [10, 10],
            [11, 11]
        ]
    ).t()

    node_type_index = [0, 9, 12]
    edge_type_index = [0, 12, 20, 24, 36]

    g = model.ComputeGraph(x, edge_index, node_type_index, edge_type_index)

    return g


if __name__ == "__main__":
    g = graph_gemm(512, 512, 512)
    out_channels = [32, 64, 128, 256, 128, 64, 32, 16, 8, 1]
    in_channel = g.in_channel
    output = g.x
    for out_channel in out_channels:
        net = model.MyConv(in_channel, out_channel, g.num_node_type, g.num_edge_type)
        output = net(output, g.node_type_index, g.edge_index, g.edge_type_index)
        in_channel = out_channel
    print(output)
    groups = list(range(g.num_nodes))
    for i in range(g.num_nodes):
        for j in range(i + 1, g.num_nodes):
            if groups[j] != j:
                continue
            if all(torch.abs(output[i] - output[j]) < 1e-5):
                groups[j] = i
    roots = set(groups)
    for id in roots:
        tmp = []
        for i in range(g.num_nodes):
            if groups[i] == id:
                tmp.append(i)
        print("Group %d: %s" % (id, str(tmp)))