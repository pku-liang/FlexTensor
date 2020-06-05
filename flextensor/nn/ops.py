"""
Common operators in deep-learning.
Follow the convention of PyTorch.

====================================
**Author**: `Size Zheng`
"""
import tvm 
import topi
from flextensor.utils import test_allclose, assert_print


def zero_pad1d(inputs, padding=0):
    """Zero padding for 1d tensor

    Args:
    -----------------------------
    inputs : tvm.te.tensor.Tensor
        shape [batch, channel, length]
    padding: (optional:0) int or tuple
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel, padded_length]
    -----------------------------
    """
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert_print(isinstance(padding, tuple), "type(padding)={}".format(type(padding)))
    assert_print(len(padding) == 2)

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, in_len = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, in_len + padding[0] + padding[1]),
        lambda b, c, l: tvm.te.if_then_else(
                            tvm.te.all(l >= padding[0], l < in_len + padding[0]),
                            inputs[b, c, l - padding[0]],
                            padding_zero
                            )
        )


def zero_expand1d(inputs, stride=1):
    """expand the inputs by zeros

    explain the expand operation:
    given stride = 1
    [1, 2, 3, 4, 5] --> expand [1, 0, 2, 0, 3, 0, 4, 0, 5]

    Args:
    -----------------------------
    inputs : tvm.te.tensor.Tensor
        shape [batch, channel, length]
    stride: (optional:0) int or tuple
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel, (length - 1) * stride + 1]
    -----------------------------
    """
    stride = stride[0] if isinstance(stride, tuple) else stride
    assert_print(isinstance(stride, (int, tvm.tir.IntImm)), "type(stride)={}".format(type(stride)))

    expand_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, in_len = inputs.shape
    out_len = (in_len - 1) * stride + 1
    return tvm.te.compute(
        (batch_size, in_channel, out_len),
        lambda b, c, l: tvm.te.if_then_else(
                            l % stride == 0,
                            inputs[b, c, l // stride],
                            expand_zero
                            )
        )


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
    assert_print(isinstance(padding, tuple), "type(padding)={}".format(type(padding)))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert_print(len(padding) == 4)

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

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


def zero_pad2d_nchwc(inputs, padding=0):
    """Zero padding for 2d tensor of NCHWc layout

    Args:
    -----------------------------
    inputs : tvm.te.tensor.Tensor
        shape [batch, channel_chunk, height, width, channel_block]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel_chunk, padded_height, padded_width, channel_block]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert_print(len(padding) == 4)

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel_chunk, height, width, in_channel_block = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel_chunk, height + padding[0] + padding[1], width + padding[2] + padding[3], in_channel_block),
        lambda b, c_c, h, w, c_b: tvm.te.if_then_else(
                            tvm.te.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                            inputs[b, c_c, h - padding[0], w - padding[2], c_b],
                            padding_zero
                            ),
        name='Padding_NCHWc'
        )


def zero_pad3d(inputs, padding=0):
    """Zero padding for 3d tensor

    Args:
    -----------------------------
    inputs : tvm.te.tensor.Tensor
        shape [batch, channel, depth, height, width]
    padding: (optional:0) int or tuple
        expected: (d_pad_up, d_pad_down, h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel, padded_depth, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert_print(isinstance(padding, tuple), "type(padding)={}".format(type(padding)))
    if len(padding) == 3:
        padding = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    assert_print(len(padding) == 6)

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel,depth, height, width = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, depth + padding[0] + padding[1], height + padding[2] + padding[3], width + padding[4] + padding[5]),
        lambda b, c, d, h, w: tvm.te.if_then_else(
                            tvm.te.all(d >= padding[0], d < depth + padding[0],h >= padding[2], h < height + padding[2], w >= padding[4], w < width + padding[4]),
                            inputs[b, c, d - padding[0], h - padding[2], w - padding[4]],
                            padding_zero
                            )
        )


def zero_expand2d(inputs, stride=1):
    """Expand the inputs by zeros

    explain the expand operation:
    given stride = 1
    [[1, 2]      [[1, 0, 2]
     [3, 4]] -->  [0, 0, 0]
                  [3, 0, 4]]

    Args:
    -----------------------------
    inputs : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    stride: (optional:0) int or tuple
        expected: (h_stride, w_stride)
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel, (height - 1) * h_stride + 1, (width - 1) * w_stride + 1]
    -----------------------------
    """
    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    assert_print(isinstance(stride, tuple), "type(stride)={}".format(type(stride)))
    assert_print(len(stride) == 2)

    expand_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    out_height = (height - 1) * stride[0] + 1
    out_width = (width - 1) * stride[1] + 1
    return tvm.te.compute(
        (batch_size, in_channel, out_height, out_width),
        lambda b, c, h, w: tvm.te.if_then_else(
                            tvm.te.all(
                                h % stride[0] == 0,
                                w % stride[1] == 0
                                ),
                            inputs[b, c, h // stride[0], w // stride[1]],
                            expand_zero
                            )
        )


def zero_expand3d(inputs, stride=1):
    """Expand the inputs by zeros

    explain the expand operation:
    given stride = 1
    [[[1, 2] --> [[[1, 0, 2]
      [3, 4]]      [0, 0, 0]                    
                   [3, 0, 4]]
     [[5, 6]        
      [7, 8]]]    [[0, 0, 0] 
                   [0, 0, 0] 
                   [0, 0, 0]]

                  [[5, 0, 6]
                   [0, 0, 0]
                   [7, 0, 8]]]

    Args:
    -----------------------------
    inputs : tvm.te.tensor.Tensor
        shape [batch, channel, depth, height, width]
    stride: (optional:0) int or tuple
        expected: (d_stride, h_stride, w_stride)
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel, (depth - 1) * d_stride + 1, (height - 1) * h_stride + 1, (width - 1) * w_stride + 1]
    -----------------------------
    """
    stride = (stride, stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    assert_print(isinstance(stride, tuple), "type(stride)={}".format(type(stride)))
    assert_print(len(stride) == 3)

    expand_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, depth, height, width = inputs.shape
    out_depth = (depth - 1) * stride[0] + 1
    out_height = (height - 1) * stride[1] + 1
    out_width = (width - 1) * stride[2] + 1
    return tvm.te.compute(
        (batch_size, in_channel, out_depth, out_height, out_width),
        lambda b, c, d ,h, w: tvm.te.if_then_else(
                            tvm.te.all(
                                d % stride[0] == 0,
                                h % stride[1] == 0,
                                w % stride[2] == 0
                                ),
                            inputs[b, c, d // stride[0], h // stride[1], w // stride[2]],
                            expand_zero
                            )
        )


def conv1d(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 1d

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, length]
    weight  : tvm.te.tensor.Tensor
        shape [out_channel, channel // groups, kernel_length]
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
        shape [batch, out_channel, output_length]
    -----------------------------
    """
    batch_size, in_channel, in_len = inputs.shape
    out_channel, channel_per_group, k_len = weight.shape
    assert_print(isinstance(groups, (int, tvm.tir.IntImm)), "type(groups)={}".format(type(groups)))
    assert_print(
        (channel_per_group * groups).value == in_channel.value, 
        "left=%d, right=%d" % (channel_per_group * groups, in_channel)
        )
    assert_print((out_channel % groups).value == 0, "left={}".format(out_channel % groups))
    out_channel_per_group = out_channel // groups

    stride = stride[0] if isinstance(stride, tuple) else stride
    padding = padding[0] if isinstance(padding, tuple) else padding
    assert_print(isinstance(stride, (int, tvm.tir.IntImm)), "type(stride)={}".format(type(stride)))
    assert_print(isinstance(padding, (int, tvm.tir.IntImm)), "type(padding)={}".format(type(padding)))
    assert_print(isinstance(dilation, (int, tvm.tir.IntImm)), "type(dilation)={}".format(type(dilation)))

    out_len = (in_len + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1

    rc = tvm.te.reduce_axis((0, channel_per_group))
    rl = tvm.te.reduce_axis((0, k_len))

    padded = zero_pad1d(inputs, padding=padding)
    conved = tvm.te.compute(
        (batch_size, out_channel, out_len),
        lambda b, c, l: tvm.te.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, l * stride + rl * dilation] * 
            weight[c, rc, rl]), 
            axis=[rc, rl]
            )
    )
    if bias is not None:
        conved = tvm.te.compute((batch_size, out_channel, out_len), lambda b, c, l: conved[b, c, l] + bias[c])
    return conved


def conv_transpose1d(inputs, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """Convolution transpose 1d

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, length]
    weight  : tvm.te.tensor.Tensor
        shape [channel, out_channel // groups, kernel_length]
    bias    : (optional:None) tvm.te.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    output_padding : (optional:0) int or tuple

    groups  : (optional:1) int

    dilation: (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, out_channel, output_length]
    -----------------------------
    """
    batch_size, input_channel, length = inputs.shape
    input_channel_w, channel_per_group, k_len = weight.shape
    assert_print(input_channel.value == input_channel_w.value)
    in_channel_per_group = input_channel // groups
    assert_print((in_channel_per_group * groups).value == input_channel.value)
    output_channel = channel_per_group * groups

    stride = stride[0] if isinstance(stride, tuple) else stride
    padding = padding[0] if isinstance(padding, tuple) else padding
    output_padding = output_padding[0] if isinstance(output_padding, tuple) else output_padding
    assert_print(isinstance(stride, (int, tvm.tir.IntImm)))
    assert_print(isinstance(padding, (int, tvm.tir.IntImm)))
    assert_print(isinstance(output_padding, (int, tvm.tir.IntImm)))
    assert_print(isinstance(groups, (int, tvm.tir.IntImm)))
    assert_print(isinstance(dilation, (int, tvm.tir.IntImm)))

    kernel_size = (k_len - 1) * dilation + 1
    output_len = (length - 1) * stride - 2 * padding + kernel_size + output_padding
    rc = tvm.te.reduce_axis((0, in_channel_per_group))
    rl = tvm.te.reduce_axis((0, k_len))

    expanded = zero_expand1d(inputs, stride=stride)
    padded = zero_pad1d(expanded, padding=(kernel_size - 1 - padding, 
                                    kernel_size - 1 - padding + output_padding))
    output = tvm.te.compute(
        (batch_size, output_channel, output_len),
        lambda b, c, l: tvm.te.sum(
            (padded[b, c // channel_per_group * in_channel_per_group + rc, l + rl * dilation] * 
            weight[c // channel_per_group * in_channel_per_group + rc, c % channel_per_group, k_len - rl - 1]),
            axis=[rc, rl])
    )
    if bias is not None:
        output = tvm.te.compute((batch_size, output_channel, output_len), lambda b, c, l: output[b, c, l] + bias[c])
    return output



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
    assert_print((channel_per_group * groups).value == in_channel.value)
    out_channel_per_group = out_channel // groups
    assert_print((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)

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


def conv2d_nchwc(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 2d NCHWc layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel // vlen1, height, width, vlen1]
    weight  : tvm.te.tensor.Tensor
        shape [out_channel // vlen2, channel // vlen1 // groups, kernel_height, kernel_width, vlen1, vlen2]
    bias    : (optional:None) tvm.te.tensor.Tensor
        shape [out_channel // vlen2, vlen2]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, out_channel // vlen2, output_height, output_width, vlen2]
    -----------------------------
    """
    batch_size, in_channel_chunk, in_h, in_w, in_channel_block = inputs.shape
    out_channel_chunk, channel_per_group_chunk, k_h, k_w, _in_channel_block, out_channel_block = weight.shape
    assert ((channel_per_group_chunk * groups).value == in_channel_chunk.value)
    assert _in_channel_block.value == in_channel_block.value
    out_channel_per_group = out_channel_chunk // groups
    assert ((out_channel_per_group * groups).value == out_channel_chunk.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert (isinstance(stride, tuple) and len(stride) == 2)
    assert (isinstance(padding, tuple) and len(padding) == 2)
    assert (isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rc_chunk = tvm.te.reduce_axis((0, channel_per_group_chunk), name="rc_chunk")
    rc_block = tvm.te.reduce_axis((0, in_channel_block), name="rc_block")
    rh = tvm.te.reduce_axis((0, k_h))
    rw = tvm.te.reduce_axis((0, k_w))

    padded = zero_pad2d_nchwc(inputs, padding=padding)
    output = tvm.te.compute(
        (batch_size, out_channel_chunk, out_h, out_w, out_channel_block),
        lambda b, c_c, h, w, c_b: tvm.te.sum(
            (padded[b, c_c // out_channel_per_group * channel_per_group_chunk + rc_chunk, 
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1], rc_block]
            * weight[c_c, rc_chunk, rh, rw, rc_block, c_b]),
            axis=[rc_chunk, rc_block, rw, rh]
        )
    )
    if bias is not None:
        output = tvm.te.compute(
            (batch_size, out_channel_chunk, out_h, out_w, out_channel_block),
            lambda b, c_c, h, w, c_b: output[b, c_c, h, w, c_b] + bias[c_c, c_b]
        )
    return output


def conv_transpose2d_nchw(inputs, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """Convolution transpose 2d NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : tvm.te.tensor.Tensor
        shape [channel, out_channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) tvm.te.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    output_padding : (optional:0) int or tuple

    groups  : (optional:1) int

    dilation: (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, input_channel, in_h, in_w = inputs.shape
    input_channel_w, channel_per_group, k_h, k_w = weight.shape
    assert_print(input_channel.value == input_channel_w.value)
    in_channel_per_group = input_channel // groups
    assert_print((in_channel_per_group * groups).value == input_channel.value)
    output_channel = channel_per_group * groups

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    output_padding = ((output_padding, output_padding) 
                        if isinstance(output_padding, (int, tvm.tir.IntImm)) else output_padding)
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(output_padding, tuple) and len(output_padding) == 2)
    assert_print(isinstance(groups, (int, tvm.tir.IntImm)))
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)

    kernel_h = (k_h - 1) * dilation[0] + 1
    kernel_w = (k_w - 1) * dilation[1] + 1
    out_h = (in_h - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    out_w = (in_w - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    rc = tvm.te.reduce_axis((0, in_channel_per_group))
    rh = tvm.te.reduce_axis((0, k_h))
    rw = tvm.te.reduce_axis((0, k_w))

    expanded = zero_expand2d(inputs, stride=stride)
    padded = zero_pad2d(expanded, padding=(
                                    kernel_h - 1 - padding[0], 
                                    kernel_h - 1 - padding[0] + output_padding[0],
                                    kernel_w - 1 - padding[1],
                                    kernel_w - 1 - padding[1] + output_padding[1]))
    output = tvm.te.compute(
        (batch_size, output_channel, out_h, out_w),
        lambda b, c, h, w: tvm.te.sum(
            (padded[b, c // channel_per_group * in_channel_per_group + rc, h + rh * dilation[0], w + rw * dilation[1]] * 
            weight[c // channel_per_group * in_channel_per_group + rc, c % channel_per_group, k_h - rh - 1, k_w - rw - 1]),
            axis=[rc, rw, rh])
    )
    if bias is not None:
        output = tvm.te.compute(
            (batch_size, output_channel, out_h, out_w), 
            lambda b, c, h, w: output[b, c, h, w] + bias[c]
            )
    return output


def depthwise_conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1):
    """Depthwise convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : tvm.te.tensor.Tensor
        shape [in_channel, factor, kernel_height, kernel_width]
    bias    : (optional:None) tvm.te.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    _in_channel, factor, k_h, k_w = weight.shape
    assert_print(_in_channel.value == in_channel.value)
    out_channel = in_channel * factor

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rh = tvm.te.reduce_axis((0, k_h))
    rw = tvm.te.reduce_axis((0, k_w))

    padded = zero_pad2d(inputs, padding=padding)
    output = tvm.te.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda b, c, h, w: tvm.te.sum(
            (padded[b, c//factor, 
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
            * weight[c//factor, c%factor, rh, rw]),
            axis=[rw, rh]
        )
    )
    if bias is not None:
        output = tvm.te.compute(
            (batch_size, out_channel, out_h, out_w),
            lambda b, c, h, w: output[b, c, h, w] + bias[c]
        )
    return output


def conv3d_ncdhw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 3d NCDHW layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, depth, height, width]
    weight  : tvm.te.tensor.Tensor
        shape [out_channel, channel // groups, kernel_depth, kernel_height, kernel_width]
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
        shape [batch, out_channel, output_depth, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_d, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_d, k_h, k_w = weight.shape
    assert_print((channel_per_group * groups).value == in_channel.value)
    out_channel_per_group = out_channel // groups
    assert_print((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 3)
    assert_print(isinstance(padding, tuple) and len(padding) == 3)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 3)

    out_d = (in_d + 2 * padding[0] - dilation[0] * (k_d - 1) - 1) // stride[0] + 1
    out_h = (in_h + 2 * padding[1] - dilation[1] * (k_h - 1) - 1) // stride[1] + 1
    out_w = (in_w + 2 * padding[2] - dilation[2] * (k_w - 1) - 1) // stride[2] + 1
    rc = tvm.te.reduce_axis((0, channel_per_group))
    rd = tvm.te.reduce_axis((0, k_d))
    rh = tvm.te.reduce_axis((0, k_h))
    rw = tvm.te.reduce_axis((0, k_w))

    padded = zero_pad3d(inputs, padding=padding)
    output = tvm.te.compute(
        (batch_size, out_channel, out_d, out_h, out_w),
        lambda b, c, d, h, w: tvm.te.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                    d * stride[0] + rd * dilation[0], h * stride[1] + rh * dilation[1], w * stride[2] + rw * dilation[2]]
            * weight[c, rc, rd, rh, rw]),
            axis=[rc, rd, rw, rh]
        )
    )
    if bias is not None:
        output = tvm.te.compute(
            (batch_size, out_channel, out_d, out_h, out_w),
            lambda b, c, d, h, w: output[b, c, d, h, w] + bias[c]
        )
    return output


def conv_transpose3d_ncdhw(inputs, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """Convolution transpose 3d NCDHW layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, depth, height, width]
    weight  : tvm.te.tensor.Tensor
        shape [channel, out_channel // groups, kernel_depth, kernel_height, kernel_width]
    bias    : (optional:None) tvm.te.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    output_padding : (optional:0) int or tuple

    groups  : (optional:1) int

    dilation: (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, out_channel, out_depth, output_height, output_width]
    -----------------------------
    """
    batch_size, input_channel, in_d, in_h, in_w = inputs.shape
    input_channel_w, channel_per_group, k_d, k_h, k_w = weight.shape
    #assert_print(input_channel.value == input_channel_w.value)
    in_channel_per_group = input_channel // groups
    #assert_print((in_channel_per_group * groups).value == input_channel.value)
    output_channel = channel_per_group * groups

    stride = (stride, stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    output_padding = ((output_padding, output_padding, output_padding) 
                        if isinstance(output_padding, (int, tvm.tir.IntImm)) else output_padding)
    dilation = (dilation, dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 3)
    assert_print(isinstance(padding, tuple) and len(padding) == 3)
    assert_print(isinstance(output_padding, tuple) and len(output_padding) == 3)
    assert_print(isinstance(groups, (int, tvm.tir.IntImm)))
    assert_print(isinstance(dilation, tuple) and len(dilation) == 3)

    kernel_d = (k_d - 1) * dilation[0] + 1
    kernel_h = (k_h - 1) * dilation[1] + 1
    kernel_w = (k_w - 1) * dilation[2] + 1
    out_d = (in_d - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0]
    out_h = (in_h - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1]
    out_w = (in_w - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2]
    rc = tvm.te.reduce_axis((0, in_channel_per_group))
    rd = tvm.te.reduce_axis((0, k_d))
    rh = tvm.te.reduce_axis((0, k_h))
    rw = tvm.te.reduce_axis((0, k_w))

    expanded = zero_expand3d(inputs, stride=stride)
    padded = zero_pad3d(expanded, padding=(
                                    kernel_d - 1 - padding[0], 
                                    kernel_d - 1 - padding[0] + output_padding[0],
                                    kernel_h - 1 - padding[1], 
                                    kernel_h - 1 - padding[1] + output_padding[1],
                                    kernel_w - 1 - padding[2],
                                    kernel_w - 1 - padding[2] + output_padding[2]))
    output = tvm.te.compute(
        (batch_size, output_channel, out_d, out_h, out_w),
        lambda b, c, d, h, w: tvm.te.sum(
            (padded[b, c // channel_per_group * in_channel_per_group + rc, d + rd * dilation[0], h + rh * dilation[1], w + rw * dilation[2]] * 
            weight[c // channel_per_group * in_channel_per_group + rc, c % channel_per_group, k_d - rd - 1, k_h - rh - 1, k_w - rw - 1]),
            axis=[rc, rd, rw, rh])
    )
    if bias is not None:
        output = tvm.te.compute(
            (batch_size, output_channel, out_d, out_h, out_w), 
            lambda b, c, d, h, w: output[b, c, d, h, w] + bias[c]
            )
    return output
    

def conv2d_nhwc(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, height, width, channel]
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
        shape [batch, output_height, output_width, out_channel]
    -----------------------------
    """
    batch_size, in_h, in_w, in_channel = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert_print((channel_per_group * groups).value == in_channel.value)
    out_channel_per_group = out_channel // groups
    assert_print((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)
    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rc = tvm.te.reduce_axis((0, channel_per_group))
    rh = tvm.te.reduce_axis((0, k_h))
    rw = tvm.te.reduce_axis((0, k_w))

    padded = zero_pad2d(inputs, padding=padding)
    output = tvm.te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda b, h, w, c: tvm.te.sum(
            (padded[b, 
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1],
                    c // out_channel_per_group * channel_per_group + rc]
            * weight[c, rc, rh, rw]),
            axis=[rc, rw, rh]
        )
    )
    if bias is not None:
        output = tvm.te.compute(
            (batch_size, out_h, out_w, out_channel),
            lambda b, h, w, c: output[b, h, w, c] + bias[c]
        )
    return output


def im2col_nchw_naive(inputs, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    """Image to column NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    kernel_size: tuple
        contains (out_channel, channel_per_group, kernel_height, kernel_width)
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [groups, height, width]
    -----------------------------
    """
    # TODO only support tvm.tir.IntImm type
    assert_print(all([isinstance(x, tvm.tir.IntImm) for x in kernel_size]))
    batch_size, channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = kernel_size
    out_channel_per_group = out_channel // groups
    assert_print((channel_per_group * groups).value == channel.value)
    assert_print((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)
    assert_print(isinstance(groups, (int, tvm.tir.IntImm)))
    
    out_height = k_h * k_w * channel_per_group
    P = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    Q = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    output_width = batch_size * P * Q

    padded = zero_pad2d(inputs,padding=padding)
    output = tvm.te.compute(
        (groups, out_height, output_width),
        lambda g, h, w: padded[w // (P * Q), 
                            g * channel_per_group + h // (k_h * k_w), 
                            w % (P * Q) // Q * stride[0] + h % (k_h * k_w) // k_w * dilation[0],
                            w % Q * stride[1] + h % k_w * dilation[1]], 
        name='Img2Col')
    return output


def col2img_nchw(inputs, P, Q):
    """Column to image NCHW layout

    Args:
    -----------------------------
    inputs: tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    P     : int or tvm.tir.IntImm
        height of output image
    Q     : int or tvm.tir.IntImm
        width of output image
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel, P, Q]
    -----------------------------
    """
    groups, out_channel_per_group, width = inputs.shape
    batch_size = width // (P * Q)
    output = tvm.te.compute(
        (batch_size, out_channel_per_group * groups, P, Q),
        lambda b, c, h, w: inputs[c // out_channel_per_group, c % out_channel_per_group, b * (P * Q) + h * Q + w],
        name='Col2Img'
    )
    return output


def gemm_conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Gemm based convolution 2D NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : tvm.te.tensor.Tensor
        shape [out_channel, channel_per_group, kernel_height, kernel_width]
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
        shape [batch, out_channel, out_height, out_width]
    -----------------------------
    """
    out_channel, _, k_h, k_w = weight.shape
    out_channel_per_group = out_channel // groups

    _, _, in_h, in_w = inputs.shape
    assert_print((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)
    assert_print(isinstance(groups, (int, tvm.tir.IntImm)))
    
    P = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    Q = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

    col = im2col_nchw_naive(inputs, weight.shape, stride, padding, dilation, groups)
    k = tvm.te.reduce_axis((0, col.shape[1]))
    gemm = tvm.te.compute(
        (groups, out_channel_per_group, col.shape[2]),
        lambda g, h, w: tvm.te.sum(col[g, k, w] * 
                                weight[g * out_channel_per_group + h, 
                                    k // (k_h * k_w), 
                                    k % (k_h * k_w) // k_w, 
                                    k % k_w],
                                axis=k
                                ), 
        name='ComputeGemm'
                    )
    if bias is not None:
        gemm = tvm.te.compute(
            (groups, out_channel_per_group, col.shape[2]),
            lambda g, h, w: gemm[g, h, w] + bias[g * out_channel_per_group + h],
            name='ComputeBias')
    img = col2img_nchw(gemm, P, Q)
    return img


def gemv(A, vector, transposeA=False):
    """Matrix multiplies vector

    Args:
    -----------------------------
    A: tvm.te.tensor.Tensor
        shape [height, width]
    vector: tvm.te.tensor.Tensor
        shape [width]
    transposeA: (optional:False) bool
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [height]
    -----------------------------
    """
    k = tvm.te.reduce_axis((0, vector.shape[0]))
    if transposeA:
        assert_print(A.shape[0].value == vector.shape[0].value)
        return tvm.te.compute((A.shape[1],), lambda i: tvm.te.sum(A[k, i] * vector[k], axis=k))
    else:
        assert_print(A.shape[1].value == vector.shape[0].value)
        return tvm.te.compute((A.shape[0],), lambda i: tvm.te.sum(A[i, k] * vector[k], axis=k))


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
        assert_print(A.shape[0].value == B.shape[1].value)
        return tvm.te.compute((A.shape[1], B.shape[0]), lambda i, j: tvm.te.sum(A[k, i] * B[j, k], axis=k))
    elif transposeA and not transposeB:
        k = tvm.te.reduce_axis((0, B.shape[0]))
        assert_print(A.shape[0].value == B.shape[0].value)
        return tvm.te.compute((A.shape[1], B.shape[1]), lambda i, j: tvm.te.sum(A[k, i] * B[k, j], axis=k))
    elif not transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[1].value == B.shape[1].value)
        return tvm.te.compute((A.shape[0], B.shape[0]), lambda i, j: tvm.te.sum(A[i, k] * B[j, k], axis=k))
    else:
        k = tvm.te.reduce_axis((0, B.shape[0]))
        assert_print(A.shape[1].value == B.shape[0].value)
        return tvm.te.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=k))


def batch_gemm(A, B, transposeA=False, transposeB=False):
    """Batched matrix multiplies matrix

    Args:
    -----------------------------
    A: tvm.te.tensor.Tensor
        shape [batch, height, width]
    B: tvm.te.tensor.Tensor
        shape [batch, width, length]
    transposeA: (optional:False) bool
    transposeB: (optional:False) bool
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, height, length]
    -----------------------------
    """
    assert_print(A.shape[0].value == B.shape[0].value)
    if transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[2]))
        assert_print(A.shape[1].value == B.shape[2].value)
        return tvm.te.compute(
            (A.shape[0], A.shape[2], B.shape[1]), 
            lambda b, i, j: tvm.te.sum(A[b, k, i] * B[b, j, k], axis=k)
            )
    elif transposeA and not transposeB:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[1].value == B.shape[1].value)
        return tvm.te.compute(
            (A.shape[0], A.shape[2], B.shape[2]), 
            lambda b, i, j: tvm.te.sum(A[b, k, i] * B[b, k, j], axis=k)
            )
    elif not transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[2]))
        assert_print(A.shape[2].value == B.shape[2].value)
        return tvm.te.compute(
            (A.shape[0], A.shape[1], B.shape[1]), 
            lambda b, i, j: tvm.te.sum(A[b, i, k] * B[b, j, k], axis=k)
            )
    else:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[2].value == B.shape[1].value)
        return tvm.te.compute(
            (A.shape[0], A.shape[1], B.shape[2]), 
            lambda b, i, j: tvm.te.sum(A[b, i, k] * B[b, k, j], axis=k)
            )


def linear(inputs, weight, bias=None):
    """Linear function

    Args:
    -----------------------------
    inputs: tvm.te.tensor.Tensor
        shape [batch, ..., in_feature]
    weight: tvm.te.tensor.Tensor
        shape [out_feature, in_feature]
    bias  : tvm.te.tensor.Tensor
        shape [out_feature]
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, ..., out_feature]
    -----------------------------
    """
    assert_print(inputs.shape[-1].value == weight.shape[1].value)
    k = tvm.te.reduce_axis((0, inputs.shape[-1]))

    def _inner(*args):
        return tvm.te.sum(inputs[(*args[:-1], k)] * weight[args[-1], k], axis=k)
    
    output = tvm.te.compute((*inputs.shape[:-1], weight.shape[0]), _inner)
    if bias is not None:
        assert_print(bias.shape[0].value == weight.shape[0].value)
        def _add(*args):
            return output[args] + bias[args[-1]]
        output = tvm.te.compute(output.shape, _add)
    return output


def bilinear(inputs1, inputs2, weight, bias=None):
    """Bilinear function

    Args:
    -----------------------------
    inputs1: tvm.te.tensor.Tensor
        shape [batch, ..., in_feature1]
    inputs2: tvm.te.tensor.Tensor
        shape [batch, ..., in_feature2]
    weight: tvm.te.tensor.Tensor
        shape [out_feature, in_feature1, in_feature2]
    bias  : tvm.te.tensor.Tensor
        shape [out_feature]
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, ..., out_feature]
    -----------------------------
    """
    assert_print(inputs1.shape[-1].value == weight.shape[-2].value)
    assert_print(inputs2.shape[-1].value == weight.shape[-1].value)
    k1 = tvm.te.reduce_axis((0, inputs1.shape[-1]))
    k2 = tvm.te.reduce_axis((0, inputs2.shape[-1]))
    for i in range(len(inputs1.shape) - 1):
        assert_print(inputs1.shape[i].value == inputs2.shape[i].value)

    def _inner(*args):
        return tvm.te.sum(
            inputs1[(*args[:-1], k1)] * weight[args[-1], k1, k2] * inputs2[(*args[:-1], k2)], 
            axis=[k1, k2]
            )
    
    output = tvm.te.compute((*inputs1.shape[:-1], weight.shape[0]), _inner)
    if bias is not None:
        assert_print(bias.shape[0].value == weight.shape[0].value)
        def _add(*args):
            return output[args] + bias[args[-1]]
        output = tvm.te.compute(output.shape, _add)
    return output


def MTTKRP3d(A, B, C):
    """Dense MTTKRP 3D

    Args:
    -----------------------------
    A: tvm.te.tensor.Tensor
        shape [out_height, h1, h2]
    B: tvm.te.tensor.Tensor
        shape [h1, out_width]
    C: tvm.te.tensor.Tensor
        shape [h2, out_width]
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [out_height, out_width]
    -----------------------------
    """
    assert_print(A.shape[1].value == B.shape[0].value)
    assert_print(A.shape[2].value == C.shape[0].value)
    assert B.shape[1].value == C.shape[1].value
    k = tvm.te.reduce_axis((0, B.shape[0]))
    l = tvm.te.reduce_axis((0, C.shape[0]))

    return tvm.te.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.te.sum(A[i, k, l] * B[k, j] * C[l, j], axis=[k, l]))


def pointwise_multiply(A, B):
    """Pointwise multiply

    Args:
    -----------------------------
    A: tvm.te.tensor.Tensor
        shape [...]
    B: tvm.te.tensor.Tensor
        shape same as A
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape same as A
    -----------------------------
    """
    assert_print(len(A.shape) == len(B.shape))
    for i in range(len(A.shape)):
        assert_print(A.shape[i].value == B.shape[i].value)
    
    def _mul(*args):
        return A[args] * B[args]

    return tvm.te.compute(A.shape, _mul)


def mean(inputs, dim=0):
    """Mean

    Args:
    -----------------------------
    A: tvm.te.tensor.Tensor
        shape [...]
    dim: (optional:0) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [...]
    -----------------------------
    """
    assert_print(dim >=0 and dim < len(inputs.shape))
    output_shape = []
    for i in range(len(inputs.shape)):
        if i == dim:
            continue
        output_shape.append(inputs.shape[i])
    
    k = tvm.te.reduce_axis((0, inputs.shape[dim]))

    def _inner(*args):
        return tvm.te.sum(inputs[(*args[:dim], k, *args[dim:])] / inputs.shape[dim], axis=k)
    return tvm.te.compute(output_shape, _inner)


def variance(inputs, mean_val=None, dim=0):
    """Variance

    Args:
    -----------------------------
    A: tvm.te.tensor.Tensor
        shape [...]
    dim: (optional:0) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [...]
    -----------------------------
    """
    assert_print(dim >= 0 and dim < len(inputs.shape))
    assert_print(inputs.shape[dim].value > 1)
    output_shape = []
    for i in range(len(inputs.shape)):
        if i == dim:
            continue
        output_shape.append(inputs.shape[i])
    
    k = tvm.te.reduce_axis((0, inputs.shape[dim]))
    mean_val = mean_val if mean_val is not None else mean(inputs, dim)

    def _inner(*args):
        return tvm.te.sum((inputs[(*args[:dim], k, *args[dim:])] - mean_val[args]) * 
                    (inputs[(*args[:dim], k, *args[dim:])] - mean_val[args]) / (inputs.shape[dim] - 1), axis=k)
    return tvm.te.compute(output_shape, _inner)


def batch_normalization2d(inputs, epsilon=1e-5):
    mean_val = mean(inputs, dim=0)
    var_val = variance(inputs, mean_val=mean_val, dim=0)
    return tvm.te.compute(inputs.shape, lambda i, j: (inputs[i, j] - mean_val[j]) / tvm.te.sqrt(var_val[j] + epsilon))


def LSTMCell(inputs, hs, cs, weights, bias=None):
    assert inputs.shape[0].value == hs.shape[0].value
    assert hs.shape[0].value == cs.shape[0].value
    assert weights.shape[0].value == 4
    assert weights.shape[2].value == inputs.shape[1].value + hs.shape[1].value
    k1 = tvm.te.reduce_axis((0, inputs.shape[1]))
    k2 = tvm.te.reduce_axis((0, hs.shape[1]))
    A = tvm.te.compute(
        (inputs.shape[0], weights.shape[0], weights.shape[1]), 
        lambda b, i, j: tvm.te.sum(inputs[b, k1] * weights[i, j, k1], axis=k1))
    B = tvm.te.compute(
        (hs.shape[0], weights.shape[0], weights.shape[1]), 
        lambda b, i, j: tvm.te.sum(hs[b, k2] * weights[i, j, k2 + inputs.shape[1]], axis=k2))
    if bias is not None:
        C = tvm.te.compute(
            (inputs.shape[0], weights.shape[0], weights.shape[1]), 
            lambda b, i, j: A[b, i, j] + B[b, i, j] + bias[b, i, j])
    else:
        C = tvm.te.compute(
            (inputs.shape[0], weights.shape[0], weights.shape[1]), 
            lambda b, i, j: A[b, i, j] + B[b, i, j])
    next_cs = tvm.te.compute(
        (cs.shape[0], weights.shape[1]), 
        lambda b, i: tvm.te.sigmoid(C[b, 1, i]) * cs[b, i] + tvm.te.sigmoid(C[b, 0, i]) * tvm.te.tanh(C[b, 3, i]))
    next_hs = tvm.te.compute(
        (hs.shape[0], weights.shape[1]), 
        lambda b, i: tvm.te.sigmoid(C[b, 2, i]) * tvm.te.tanh(next_cs[b, i]))
    return next_hs, next_cs


def block_circulant_matrix(Input, factor):
    ROW, COL = Input.shape
    FFT = factor

    k = tvm.te.reduce_axis((0, FFT))
    Compress = tvm.te.compute(
        (ROW // FFT, (COL // FFT) * FFT),
        lambda i, j: (
            tvm.te.sum(
                Input[i * FFT + k, (j // FFT) * FFT + (j % FFT + k) % FFT] / FFT,
                axis=k
            )
        )
    )

    Output = tvm.te.compute(
        (ROW, COL),
        lambda i, j: (
            tvm.te.if_then_else(
                tvm.te.all(i < (ROW // FFT) * FFT, j < (COL // FFT) * FFT),
                Compress[i // FFT, (j // FFT) * FFT + ((j % FFT) + FFT - (i % FFT)) % FFT], 
                tvm.tir.const(0, Input.dtype)
            )
        )
    )

    return Output


def MaxUnpooling1d(Input, Indices, kernel_size, stride, padding):
    """
    Max Unpooling 1d Operator

    Parameters
    ----------
    Input: tvm.te.tensor.Tensor
        3-D with shape [batch_size, channels, in_lengths]
    Indices: tvm.te.tensor.Tensor
        3-D with shape [batch_size, channels, out_lengths]
    kernel_size: int
    stride: int

    Returns
    -------
    Output: tvm.te.tensor.Tensor
        3-D with shape [batch_size, channels, out_lengths]
    """

    batch_size, channels, in_lengths = Input.shape
    batch_size, channels, in_lengths = Indices.shape

    out_lengths = (in_lengths - 1) * stride - 2 * padding + kernel_size

    iterK = tvm.te.reduce_axis((0, in_lengths), name='k')

    Output = tvm.te.compute((batch_size, channels, out_lengths), 
                          lambda b, c, l : 
                            tvm.te.max(
                                tvm.te.if_then_else(l == Indices[b, c, iterK], 
                                                 Input[b, c, iterK], 
                                                 tvm.tir.expr.const(0, Input.dtype)), 
                                axis=iterK), 
                          name='output')

    return Output

def MaxUnpooling2d(Input, Indices, kernel_size, stride, padding, output_size=None):
    """
    Max Unpooling 2d Operator

    Parameters
    ----------
    Input: tvm.te.tensor.Tensor
        4-D with shape [batch_size, channels, in_height, in_width]
    Indices: tvm.te.tensor.Tensor
        4-D with shape [batch_size, channels, in_height, in_width]
    kernel_size: int or tuple
    stride: int or tuple

    Returns
    -------
    Output: tvm.te.tensor.Tensor
        4-D with shape [batch_size, channels, out_height, out_width]
    """

    batch_size, channels, in_height, in_width = Input.shape
    batch_size, channels, in_height, in_width = Indices.shape

    if type(kernel_size) == int:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) == int:
        stride = (stride, stride)
    if type(padding) == int:
        padding = (padding, padding)

    out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
    out_width = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_size[1]

    iterH = tvm.te.reduce_axis((0, in_height), name='h')
    iterW = tvm.te.reduce_axis((0, in_width), name='ws')

    Output = tvm.te.compute((batch_size, channels, out_height, out_width), 
                          lambda b, c, h, w : 
                            tvm.te.max(
                                tvm.te.if_then_else(h * out_width + w == Indices[b, c, iterH, iterW], 
                                                 Input[b, c, iterH, iterW], 
                                                 tvm.tir.expr.const(0, Input.dtype)), 
                                axis=[iterH, iterW]), 
                          name='output')
    return Output

def _ShiftConv2d_nhwc(Input, Kernel, dilation, stride):
    """
    Shift Convolution Operator

    Parameters
    ----------
    Input: tvm.te.tensor.Tensor
        4-D with shape [batch_size, input_height, input_width, channels]
    Kernel: tvm.te.tensor.Tensor
        4-D with shape [channels, kernel_height, kernel_width]
    dilation: int or tuple
    stride: int or tuple

    Returns
    -------
    Output: tvm.te.tensor.Tensor
        4-D with shape [batch_size, out_height, out_width, channels]
    """

    batch, inputHeight, inputWidth, channels = Input.shape
    channels_, kernelHeight, kernelWidth = Kernel.shape

    assert channels.value == channels_.value

    if type(dilation) == int:
        dilation = (dilation, dilation)
    if type(stride) == int:
        stride = (stride, stride)
    
    assert len(dilation) == 2
    assert len(stride) == 2

    padding = [((stride[0] - 1) * inputHeight - stride[0] + dilation[0] * (kernelHeight - 1) + 1) / 2, 
                ((stride[1] - 1) * inputWidth - stride[1] + dilation[1] * (kernelWidth - 1) + 1) / 2]

    outHeight = (inputHeight + 2 * padding[0]- dilation[0] * (kernelHeight - 1) - 1) // stride[0] + 1
    outWidth = (inputWidth + 2 * padding[1] - dilation[1] * (kernelWidth - 1) - 1) // stride[1] + 1

    PInput = topi.nn.pad(Input, (0, padding[0], padding[1], 0),
                            (0, padding[0], padding[1], 0), name="PInput")

    # argmax(data, axis=None, keepdims=False): topi argmax function
    kernelIndex = topi.argmax(Kernel, axis=(1, 2))

    Output = tvm.te.compute((batch, outHeight, outWidth, channels),
                     lambda n, h, w, o : PInput[n, h * stride[0] + (kernelIndex[o] // kernelHeight) * dilation[0], 
                                                   w * stride[1] + (kernelIndex[o] % kernelWidth) * dilation[1], 
                                                o],
                     name="Output")
    # return PInput, kernelIndex, Output
    return Output


def ShiftConv2d_nhwc(Input, KernelIndex, KernelShape, dilation, stride):
    """
    Shift Convolution Operator

    Parameters
    ----------
    Input: tvm.te.tensor.Tensor
        4-D with shape [batch_size, input_height, input_width, channels]
    KernelIndex: tvm.te.tensor.Tensor
        1-D with shape [channels] integers ranging in [0, kernel_height * kernel_width)
    KernelShape: int or tuple, specify kernel height and width
    dilation: int or tuple
    stride: int or tuple

    Returns
    -------
    Output: tvm.te.tensor.Tensor
        4-D with shape [batch_size, out_height, out_width, channels]
    """

    batch, inputHeight, inputWidth, channels = Input.shape
    # channels_, kernelHeight, kernelWidth = Kernel.shape
    channels_ = KernelIndex.shape[0]
    if isinstance(KernelShape, int):
        kernelHeight, kernelWidth = KernelShape, KernelShape
    else:
        assert isinstance(KernelShape, tuple) and len(KernelShape) == 2 and isinstance(KernelShape[0], int)
        kernelHeight, kernelWidth = KernelShape

    assert channels.value == channels_.value

    if type(dilation) == int:
        dilation = (dilation, dilation)
    if type(stride) == int:
        stride = (stride, stride)
    
    assert len(dilation) == 2
    assert len(stride) == 2

    padding = [((stride[0] - 1) * inputHeight - stride[0] + dilation[0] * (kernelHeight - 1) + 1) / 2, 
                ((stride[1] - 1) * inputWidth - stride[1] + dilation[1] * (kernelWidth - 1) + 1) / 2]

    outHeight = (inputHeight + 2 * padding[0]- dilation[0] * (kernelHeight - 1) - 1) // stride[0] + 1
    outWidth = (inputWidth + 2 * padding[1] - dilation[1] * (kernelWidth - 1) - 1) // stride[1] + 1

    PInput = topi.nn.pad(Input, (0, padding[0], padding[1], 0),
                            (0, padding[0], padding[1], 0), name="PInput")

    # argmax(data, axis=None, keepdims=False): topi argmax function
    # kernelIndex = topi.argmax(Kernel, axis=(1, 2))

    Output = tvm.te.compute((batch, outHeight, outWidth, channels),
                     lambda n, h, w, o : PInput[n, h * stride[0] + (KernelIndex[o] // kernelHeight) * dilation[0], 
                                                   w * stride[1] + (KernelIndex[o] % kernelWidth) * dilation[1], 
                                                o],
                     name="Output")
    # return PInput, kernelIndex, Output
    return Output


def PixelCNN(Input, Kernel, mask_type, bias=None, dilation=1, stride=1, padding=0):
    """
    Pixel CNN Operator

    Parameters
    ----------
    Input: tvm.te.tensor.Tensor
        4-D with shape [batch_size, input_height, input_width, in_channels]
    Kernel: tvm.te.tensor.Tensor
        4-D with shape [out_channels, in_channels, kernel_height, kernel_width]
    mask_type: str 'A' or 'B'
    dilation: int or tuple
    stride: int or tuple
    padding: int or tuple

    Returns
    -------
    Output: tvm.te.tensor.Tensor
        4-D with shape [batch_size, out_height, out_width, channels]
    """

    batch, inputHeight, inputWidth, in_channels = Input.shape
    batch, out_channels, kernelHeight, kernelWidth = Kernel.shape

    assert mask_type in ['A', 'B']

    if mask_type == 'A':
        Mask = tvm.te.compute(Kernel.shape, 
                           lambda b, o, h, w : tvm.te.if_then_else(tvm.tir.Or(tvm.tir.And(h == kernelHeight // 2, w >= kernelWidth // 2), h > kernelHeight // 2), tvm.tir.expr.const(0, Input.dtype), Kernel[b, o, h, w]), 
                           name='MaskA')
    else:
        Mask = tvm.te.compute(Kernel.shape, 
                           lambda b, o, h, w : tvm.te.if_then_else(tvm.tir.Or(tvm.tir.And(h == kernelHeight // 2, w > kernelWidth // 2), h > kernelHeight // 2), tvm.tir.expr.const(0, Input.dtype), Kernel[b, o, h, w]), 
                           name='MaskB')
    
    Output = conv2d_nhwc(Input, Mask, bias, stride=stride, padding=padding, dilation=dilation)

    return Mask, Output


def GatedPixelCNN(Input, KernelV, KernelV2H, KernelH, KernelHOut, ClassVector=None, bias=None, dilation=1, stride=1, padding=0):
    """
    Gated Pixel CNN Operator

    Parameters
    ----------
    Input: tvm.te.tensor.Tensor
        4-D with shape [batch_size, input_height, input_width, in_channels]
    KernelV: tvm.te.tensor.Tensor
        Vertical Kernel
        4-D with shape [2 * out_channels, in_channels, kernel_size, kernel_size]
    KernelV2H: tvm.te.tensor.Tensor
        Combine output from vertical to horizontal
        4-D with shape [2 * out_channels, 2 * out_channels, 1, 1]
    KernelH: tvm.te.tensor.Tensor
        Horizontal Kernel
        4-D with shape [2 * out_channels, in_channels, 1, kernel_size]
    KernelHOut: tvm.te.tensor.Tensor
        Horizontal Output Kernel
        4-D with shape [out_channels, out_channels, 1, 1]
    ClassVector: tvm.te.tensor.Tensor
        4-D with shape [batch_size, 2 * out_channels, 1, 1]
    dilation: int
    stride: int
    padding: int

    Returns
    -------
    GateV: tvm.te.tensor.Tensor
        4-D with shape [batch_szie, out_height, out_width, out_channels]
    Output: tvm.te.tensor.Tensor
        4-D with shape [batch_size, out_height, out_width, out_channels]
    """
    batch, inputHeight, inputWidth, in_channels = Input.shape
    out_channels, in_channels, kernelHeight, kernelWidth = KernelV.shape
    out_channels /= 2

    assert kernelHeight.value == kernelWidth.value

    ConvV = PixelCNN(Input, KernelV, mask_type='B', bias=bias, dilation=(dilation, dilation), stride=(stride, stride), padding=(padding, padding))[-1]
    Vertical2HorizonTal = conv2d_nhwc(ConvV, KernelV2H, bias=bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1))
    ConvH = PixelCNN(Input, KernelH, mask_type='B', bias=bias, dilation=(dilation, dilation), stride=(stride, stride), padding=(0, padding))[-1]
    CombineFeature = tvm.te.compute(ConvH.shape, 
                                 lambda b, h, w, c : ConvH[b, h, w, c] + Vertical2HorizonTal[b, h, w, c], 
                                 name='CombineFeature')
    if ClassVector == None:
        ActivationV = tvm.te.compute(ConvV.shape, 
                            lambda b, h, w, o : tvm.te.if_then_else(o < out_channels, 
                                                                 tvm.te.tanh(ConvV[b, h, w, o]), 
                                                                 tvm.te.sigmoid(ConvV[b, h, w, o])), 
                            name="ActivationV")
    else:
        ActivationV = tvm.te.compute(ConvV.shape, 
                            lambda b, h, w, o : tvm.te.if_then_else(o < out_channels, 
                                                                 tvm.te.tanh(ConvV[b, h, w, o] + ClassVector[b, 0, 0, o]), 
                                                                 tvm.te.sigmoid(ConvV[b, h, w, o] + ClassVector[b, 0, 0, o])), 
                            name='ActivationV')
    GateV = tvm.te.compute((batch, ActivationV.shape[1], ActivationV.shape[2], out_channels), 
                         lambda b, h, w, c : ActivationV[b, h, w, c] * ActivationV[b, h, w, c + out_channels], 
                         name='GateV')
    
    ActivationH = tvm.te.compute(CombineFeature.shape, 
                              lambda b, h, w, o : tvm.te.if_then_else(o < out_channels, 
                                                                   tvm.te.tanh(CombineFeature[b, h, w, o]), 
                                                                   tvm.te.sigmoid(CombineFeature[b, h, w, o])), 
                              name="ActivationH")
    GateH = tvm.te.compute((batch, ActivationH.shape[1], ActivationH.shape[2], out_channels), 
                         lambda b, h, w, c : ActivationH[b, h, w, c] * ActivationH[b, h, w, c + out_channels], 
                         name='GateH')
    ConvGateH = conv2d_nhwc(GateH, KernelHOut, bias=bias, dilation=(dilation, dilation), stride=(stride, stride), padding=(padding, padding))
    Output = tvm.te.compute(ConvGateH.shape, 
                         lambda b, h, w, o : ConvGateH[b, h, w, o] + Input[b, h, w, o], 
                         name='Output')
    return GateV, Output