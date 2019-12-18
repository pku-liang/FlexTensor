import tvm
import torch
import torch_geometric


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


def schedule_gemm_cuda(op, config):
    s = tvm.create_schedule(op)
    A = op.input_tensors[0]
    B = op.input_tensors[1]
    i, j = s[op].op.axis
    bi, ii = s[op].split(i, nparts=config["spatial"][0][0])
    vi, ii = s[op].split(ii, nparts=config["spatial"][0][1])
    ti, ii = s[op].split(ii, nparts=config["spatial"][0][2])
    bj, jj = s[op].split(j, nparts=config["spatial"][1][0])
    vj, jj = s[op].split(jj, nparts=config["spatial"][1][1])
    tj, jj = s[op].split(jj, naprts=config["spatial"][1][2])

    s[op].reorder(bi, bj, vi, vj, ti, tj, ii, jj)

    write_cache = s.cache_write(op.output(0), "local")
    s[write_cache].compute_at(s[op], tj)
    AA = s.cache_read()



def graph_gemm(M, N, K):
    pass