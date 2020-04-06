import tvm


def gemm(A, B, transposeA=False, transposeB=False):
    """Matrix multiplies matrix

    Args:
    -----------------------------
    A: tvm.te.tensor.Tensor
        shape [height, width]
    B: tvm.te.tensor.Tensor
        shape [width, length]
    transposeA: (optional:False) bool
    transposeB: (optional:False) bool
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [height, length]
    -----------------------------
    """
    if transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert (A.shape[0].value == B.shape[1].value)
        return tvm.te.compute((A.shape[1], B.shape[0]), lambda i, j: tvm.te.sum(A[k, i] * B[j, k], axis=k))
    elif transposeA and not transposeB:
        k = tvm.te.reduce_axis((0, B.shape[0]))
        assert (A.shape[0].value == B.shape[0].value)
        return tvm.te.compute((A.shape[1], B.shape[1]), lambda i, j: tvm.te.sum(A[k, i] * B[k, j], axis=k))
    elif not transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert (A.shape[1].value == B.shape[1].value)
        return tvm.te.compute((A.shape[0], B.shape[0]), lambda i, j: tvm.te.sum(A[i, k] * B[j, k], axis=k))
    else:
        k = tvm.te.reduce_axis((0, B.shape[0]))
        assert (A.shape[1].value == B.shape[0].value)
        return tvm.te.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=k))


def zero_pad2d(inputs, padding=0):
    """Zero padding for 2d tensor

    Args:
    -----------------------------
    inputs : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple)
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert (len(padding) == 4)

    padding_zero = 0.0 if "float" in inputs.dtype else 0

    batch_size, in_channel, height, width = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.te.if_then_else(
                            tvm.te.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                            inputs[b, c, h - padding[0], w - padding[2]],
                            padding_zero
                            ),
        name='Padding'
        )


def conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : tvm.te.tensor.Tensor
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) tvm.te.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert ((channel_per_group * groups).value == in_channel.value)
    out_channel_per_group = out_channel // groups
    assert ((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert (isinstance(stride, tuple) and len(stride) == 2)
    assert (isinstance(padding, tuple) and len(padding) == 2)
    assert (isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")

    padded = zero_pad2d(inputs, padding=padding)
    output = tvm.te.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda b, c, h, w: tvm.te.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
            * weight[c, rc, rh, rw]),
            axis=[rc, rw, rh]
        )
    )
    if bias is not None:
        output = tvm.te.compute(
            (batch_size, out_channel, out_h, out_w),
            lambda b, c, h, w: output[b, c, h, w] + bias[c]
        )
    return output


def GEMM(M, K, N):
    A = tvm.te.placeholder((M, K), name="A")
    B = tvm.te.placeholder((K, N), name="B")
    C = gemm(A, B)
    return [C.op], [A, B, C]


def Conv2d(N, C, H, W, K, k, stride=1, padding=0, dilation=1, groups=1):
    A = tvm.te.placeholder((N, C, H, W), name="A")
    B = tvm.te.placeholder((K, C, k, k), name="B")
    C = conv2d_nchw(A, B, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return [C.op], [A, B, C]