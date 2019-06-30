"""
Common operators in deep-learning.
Follow the convention of PyTorch.

====================================
**Author**: `Size Zheng`
"""
import tvm 
from auto_schedule.utils import test_allclose, assert_print


def zero_pad1d(inputs, padding=0):
    """Zero padding for 1d tensor

    Args:
    -----------------------------
    inputs : tvm.tensor.Tensor
        shape [batch, channel, length]
    padding: (optional:0) int or tuple
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, channel, padded_length]
    -----------------------------
    """
    padding = (padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    assert_print(isinstance(padding, tuple), "type(padding)={}".format(type(padding)))
    assert_print(len(padding) == 2)

    padding_zero = 0.0 if "float" in inputs.dtype else 0

    batch_size, in_channel, in_len = inputs.shape
    return tvm.compute(
        (batch_size, in_channel, in_len + padding[0] + padding[1]),
        lambda b, c, l: tvm.if_then_else(
                            tvm.all(l >= padding[0], l < in_len + padding[0]),
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
    inputs : tvm.tensor.Tensor
        shape [batch, channel, length]
    stride: (optional:0) int or tuple
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, channel, (length - 1) * stride + 1]
    -----------------------------
    """
    stride = stride[0] if isinstance(stride, tuple) else stride
    assert_print(isinstance(stride, (int, tvm.expr.IntImm)), "type(stride)={}".format(type(stride)))

    expand_zero = 0.0 if "float" in inputs.dtype else 0

    batch_size, in_channel, in_len = inputs.shape
    out_len = (in_len - 1) * stride + 1
    return tvm.compute(
        (batch_size, in_channel, out_len),
        lambda b, c, l: tvm.if_then_else(
                            l % stride == 0,
                            inputs[b, c, l // stride],
                            expand_zero
                            )
        )


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
    assert_print(isinstance(padding, tuple), "type(padding)={}".format(type(padding)))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert_print(len(padding) == 4)

    padding_zero = 0.0 if "float" in inputs.dtype else 0

    batch_size, in_channel, height, width = inputs.shape
    return tvm.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.if_then_else(
                            tvm.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                            inputs[b, c, h - padding[0], w - padding[2]],
                            padding_zero
                            )
        )


def zero_pad3d(inputs, padding=0):
    """Zero padding for 3d tensor

    Args:
    -----------------------------
    inputs : tvm.tensor.Tensor
        shape [batch, channel, depth, height, width]
    padding: (optional:0) int or tuple
        expected: (d_pad_up, d_pad_down, h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, channel, padded_depth, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding, padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    assert_print(isinstance(padding, tuple), "type(padding)={}".format(type(padding)))
    if len(padding) == 3:
        padding = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    assert_print(len(padding) == 6)

    padding_zero = 0.0 if "float" in inputs.dtype else 0

    batch_size, in_channel,depth, height, width = inputs.shape
    return tvm.compute(
        (batch_size, in_channel, depth + padding[0] + padding[1], height + padding[2] + padding[3], width + padding[4] + padding[5]),
        lambda b, c, d, h, w: tvm.if_then_else(
                            tvm.all(d >= padding[0], d < depth + padding[0],h >= padding[2], h < height + padding[2], w >= padding[4], w < width + padding[4]),
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
    inputs : tvm.tensor.Tensor
        shape [batch, channel, height, width]
    stride: (optional:0) int or tuple
        expected: (h_stride, w_stride)
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, channel, (height - 1) * h_stride + 1, (width - 1) * w_stride + 1]
    -----------------------------
    """
    stride = (stride, stride) if isinstance(stride, (int, tvm.expr.IntImm)) else stride
    assert_print(isinstance(stride, tuple), "type(stride)={}".format(type(stride)))
    assert_print(len(stride) == 2)

    expand_zero = 0.0 if "float" in inputs.dtype else 0

    batch_size, in_channel, height, width = inputs.shape
    out_height = (height - 1) * stride[0] + 1
    out_width = (width - 1) * stride[1] + 1
    return tvm.compute(
        (batch_size, in_channel, out_height, out_width),
        lambda b, c, h, w: tvm.if_then_else(
                            tvm.all(
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
    inputs : tvm.tensor.Tensor
        shape [batch, channel, depth, height, width]
    stride: (optional:0) int or tuple
        expected: (d_stride, h_stride, w_stride)
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, channel, (depth - 1) * d_stride + 1, (height - 1) * h_stride + 1, (width - 1) * w_stride + 1]
    -----------------------------
    """
    stride = (stride, stride, stride) if isinstance(stride, (int, tvm.expr.IntImm)) else stride
    assert_print(isinstance(stride, tuple), "type(stride)={}".format(type(stride)))
    assert_print(len(stride) == 3)

    expand_zero = 0.0 if "float" in inputs.dtype else 0

    batch_size, in_channel, depth, height, width = inputs.shape
    out_depth = (depth - 1) * stride[0] + 1
    out_height = (height - 1) * stride[1] + 1
    out_width = (width - 1) * stride[2] + 1
    return tvm.compute(
        (batch_size, in_channel, out_depth, out_height, out_width),
        lambda b, c, d ,h, w: tvm.if_then_else(
                            tvm.all(
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
    inputs  : tvm.tensor.Tensor
        shape [batch, channel, length]
    weight  : tvm.tensor.Tensor
        shape [out_channel, channel // groups, kernel_length]
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
        shape [batch, out_channel, output_length]
    -----------------------------
    """
    batch_size, in_channel, in_len = inputs.shape
    out_channel, channel_per_group, k_len = weight.shape
    assert_print(isinstance(groups, (int, tvm.expr.IntImm)), "type(groups)={}".format(type(groups)))
    assert_print(
        (channel_per_group * groups).value == in_channel.value, 
        "left=%d, right=%d" % (channel_per_group * groups, in_channel)
        )
    assert_print((out_channel % groups).value == 0, "left={}".format(out_channel % groups))
    out_channel_per_group = out_channel // groups

    stride = stride[0] if isinstance(stride, tuple) else stride
    padding = padding[0] if isinstance(padding, tuple) else padding
    assert_print(isinstance(stride, (int, tvm.expr.IntImm)), "type(stride)={}".format(type(stride)))
    assert_print(isinstance(padding, (int, tvm.expr.IntImm)), "type(padding)={}".format(type(padding)))
    assert_print(isinstance(dilation, (int, tvm.expr.IntImm)), "type(dilation)={}".format(type(dilation)))

    out_len = (in_len + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1

    rc = tvm.reduce_axis((0, channel_per_group))
    rl = tvm.reduce_axis((0, k_len))

    padded = zero_pad1d(inputs, padding=padding)
    conved = tvm.compute(
        (batch_size, out_channel, out_len),
        lambda b, c, l: tvm.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, l * stride + rl * dilation] * 
            weight[c, rc, rl]), 
            axis=[rc, rl]
            )
    )
    if bias is not None:
        conved = tvm.compute((batch_size, out_channel, out_len), lambda b, c, l: conved[b, c, l] + bias[c])
    return conved


def conv_transpose1d(inputs, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """Convolution transpose 1d

    Args:
    -----------------------------
    inputs  : tvm.tensor.Tensor
        shape [batch, channel, length]
    weight  : tvm.tensor.Tensor
        shape [channel, out_channel // groups, kernel_length]
    bias    : (optional:None) tvm.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    output_padding : (optional:0) int or tuple

    groups  : (optional:1) int

    dilation: (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
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
    assert_print(isinstance(stride, (int, tvm.expr.IntImm)))
    assert_print(isinstance(padding, (int, tvm.expr.IntImm)))
    assert_print(isinstance(output_padding, (int, tvm.expr.IntImm)))
    assert_print(isinstance(groups, (int, tvm.expr.IntImm)))
    assert_print(isinstance(dilation, (int, tvm.expr.IntImm)))

    kernel_size = (k_len - 1) * dilation + 1
    output_len = (length - 1) * stride - 2 * padding + kernel_size + output_padding
    rc = tvm.reduce_axis((0, in_channel_per_group))
    rl = tvm.reduce_axis((0, k_len))

    expanded = zero_expand1d(inputs, stride=stride)
    padded = zero_pad1d(expanded, padding=(kernel_size - 1 - padding, 
                                    kernel_size - 1 - padding + output_padding))
    output = tvm.compute(
        (batch_size, output_channel, output_len),
        lambda b, c, l: tvm.sum(
            (padded[b, c // channel_per_group * in_channel_per_group + rc, l + rl * dilation] * 
            weight[c // channel_per_group * in_channel_per_group + rc, c % channel_per_group, k_len - rl - 1]),
            axis=[rc, rl])
    )
    if bias is not None:
        output = tvm.compute((batch_size, output_channel, output_len), lambda b, c, l: output[b, c, l] + bias[c])
    return output



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
    assert_print((channel_per_group * groups).value == in_channel.value)
    out_channel_per_group = out_channel // groups
    assert_print((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.expr.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.expr.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rc = tvm.reduce_axis((0, channel_per_group))
    rh = tvm.reduce_axis((0, k_h))
    rw = tvm.reduce_axis((0, k_w))

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


def conv_transpose2d_nchw(inputs, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """Convolution transpose 2d NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : tvm.tensor.Tensor
        shape [channel, out_channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) tvm.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    output_padding : (optional:0) int or tuple

    groups  : (optional:1) int

    dilation: (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, input_channel, in_h, in_w = inputs.shape
    input_channel_w, channel_per_group, k_h, k_w = weight.shape
    assert_print(input_channel.value == input_channel_w.value)
    in_channel_per_group = input_channel // groups
    assert_print((in_channel_per_group * groups).value == input_channel.value)
    output_channel = channel_per_group * groups

    stride = (stride, stride) if isinstance(stride, (int, tvm.expr.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    output_padding = ((output_padding, output_padding) 
                        if isinstance(output_padding, (int, tvm.expr.IntImm)) else output_padding)
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.expr.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(output_padding, tuple) and len(output_padding) == 2)
    assert_print(isinstance(groups, (int, tvm.expr.IntImm)))
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)

    kernel_h = (k_h - 1) * dilation[0] + 1
    kernel_w = (k_w - 1) * dilation[1] + 1
    out_h = (in_h - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    out_w = (in_w - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    rc = tvm.reduce_axis((0, in_channel_per_group))
    rh = tvm.reduce_axis((0, k_h))
    rw = tvm.reduce_axis((0, k_w))

    expanded = zero_expand2d(inputs, stride=stride)
    padded = zero_pad2d(expanded, padding=(
                                    kernel_h - 1 - padding[0], 
                                    kernel_h - 1 - padding[0] + output_padding[0],
                                    kernel_w - 1 - padding[1],
                                    kernel_w - 1 - padding[1] + output_padding[1]))
    output = tvm.compute(
        (batch_size, output_channel, out_h, out_w),
        lambda b, c, h, w: tvm.sum(
            (padded[b, c // channel_per_group * in_channel_per_group + rc, h + rh * dilation[0], w + rw * dilation[1]] * 
            weight[c // channel_per_group * in_channel_per_group + rc, c % channel_per_group, k_h - rh - 1, k_w - rw - 1]),
            axis=[rc, rw, rh])
    )
    if bias is not None:
        output = tvm.compute(
            (batch_size, output_channel, out_h, out_w), 
            lambda b, c, h, w: output[b, c, h, w] + bias[c]
            )
    return output


def depthwise_conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1):
    """Depthwise convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : tvm.tensor.Tensor
        shape [in_channel, factor, kernel_height, kernel_width]
    bias    : (optional:None) tvm.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    _in_channel, factor, k_h, k_w = weight.shape
    assert_print(_in_channel.value == in_channel.value)
    out_channel = in_channel * factor

    stride = (stride, stride) if isinstance(stride, (int, tvm.expr.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.expr.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rh = tvm.reduce_axis((0, k_h))
    rw = tvm.reduce_axis((0, k_w))

    padded = zero_pad2d(inputs, padding=padding)
    output = tvm.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda b, c, h, w: tvm.sum(
            (padded[b, c//factor, 
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
            * weight[c//factor, c%factor, rh, rw]),
            axis=[rw, rh]
        )
    )
    if bias is not None:
        output = tvm.compute(
            (batch_size, out_channel, out_h, out_w),
            lambda b, c, h, w: output[b, c, h, w] + bias[c]
        )
    return output


def conv3d_ncdhw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 3d NCDHW layout

    Args:
    -----------------------------
    inputs  : tvm.tensor.Tensor
        shape [batch, channel, depth, height, width]
    weight  : tvm.tensor.Tensor
        shape [out_channel, channel // groups, kernel_depth, kernel_height, kernel_width]
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
        shape [batch, out_channel, output_depth, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_d, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_d, k_h, k_w = weight.shape
    assert_print((channel_per_group * groups).value == in_channel.value)
    out_channel_per_group = out_channel // groups
    assert_print((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride, stride) if isinstance(stride, (int, tvm.expr.IntImm)) else stride
    padding = (padding, padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    dilation = (dilation, dilation, dilation) if isinstance(dilation, (int, tvm.expr.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 3)
    assert_print(isinstance(padding, tuple) and len(padding) == 3)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 3)

    out_d = (in_d + 2 * padding[0] - dilation[0] * (k_d - 1) - 1) // stride[0] + 1
    out_h = (in_h + 2 * padding[1] - dilation[1] * (k_h - 1) - 1) // stride[1] + 1
    out_w = (in_w + 2 * padding[2] - dilation[2] * (k_w - 1) - 1) // stride[2] + 1
    rc = tvm.reduce_axis((0, channel_per_group))
    rd = tvm.reduce_axis((0, k_d))
    rh = tvm.reduce_axis((0, k_h))
    rw = tvm.reduce_axis((0, k_w))

    padded = zero_pad3d(inputs, padding=padding)
    output = tvm.compute(
        (batch_size, out_channel, out_d, out_h, out_w),
        lambda b, c, d, h, w: tvm.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                    d * stride[0] + rd * dilation[0], h * stride[1] + rh * dilation[1], w * stride[2] + rw * dilation[2]]
            * weight[c, rc, rd, rh, rw]),
            axis=[rc, rd, rw, rh]
        )
    )
    if bias is not None:
        output = tvm.compute(
            (batch_size, out_channel, out_d, out_h, out_w),
            lambda b, c, d, h, w: output[b, c, d, h, w] + bias[c]
        )
    return output


def conv_transpose3d_ncdhw(inputs, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """Convolution transpose 3d NCDHW layout

    Args:
    -----------------------------
    inputs  : tvm.tensor.Tensor
        shape [batch, channel, depth, height, width]
    weight  : tvm.tensor.Tensor
        shape [channel, out_channel // groups, kernel_depth, kernel_height, kernel_width]
    bias    : (optional:None) tvm.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    output_padding : (optional:0) int or tuple

    groups  : (optional:1) int

    dilation: (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, out_channel, out_depth, output_height, output_width]
    -----------------------------
    """
    batch_size, input_channel, in_d, in_h, in_w = inputs.shape
    input_channel_w, channel_per_group, k_d, k_h, k_w = weight.shape
    #assert_print(input_channel.value == input_channel_w.value)
    in_channel_per_group = input_channel // groups
    #assert_print((in_channel_per_group * groups).value == input_channel.value)
    output_channel = channel_per_group * groups

    stride = (stride, stride, stride) if isinstance(stride, (int, tvm.expr.IntImm)) else stride
    padding = (padding, padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    output_padding = ((output_padding, output_padding, output_padding) 
                        if isinstance(output_padding, (int, tvm.expr.IntImm)) else output_padding)
    dilation = (dilation, dilation, dilation) if isinstance(dilation, (int, tvm.expr.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 3)
    assert_print(isinstance(padding, tuple) and len(padding) == 3)
    assert_print(isinstance(output_padding, tuple) and len(output_padding) == 3)
    assert_print(isinstance(groups, (int, tvm.expr.IntImm)))
    assert_print(isinstance(dilation, tuple) and len(dilation) == 3)

    kernel_d = (k_d - 1) * dilation[0] + 1
    kernel_h = (k_h - 1) * dilation[1] + 1
    kernel_w = (k_w - 1) * dilation[2] + 1
    out_d = (in_d - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0]
    out_h = (in_h - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1]
    out_w = (in_w - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2]
    rc = tvm.reduce_axis((0, in_channel_per_group))
    rd = tvm.reduce_axis((0, k_d))
    rh = tvm.reduce_axis((0, k_h))
    rw = tvm.reduce_axis((0, k_w))

    expanded = zero_expand3d(inputs, stride=stride)
    padded = zero_pad3d(expanded, padding=(
                                    kernel_d - 1 - padding[0], 
                                    kernel_d - 1 - padding[0] + output_padding[0],
                                    kernel_h - 1 - padding[1], 
                                    kernel_h - 1 - padding[1] + output_padding[1],
                                    kernel_w - 1 - padding[2],
                                    kernel_w - 1 - padding[2] + output_padding[2]))
    output = tvm.compute(
        (batch_size, output_channel, out_d, out_h, out_w),
        lambda b, c, d, h, w: tvm.sum(
            (padded[b, c // channel_per_group * in_channel_per_group + rc, d + rd * dilation[0], h + rh * dilation[1], w + rw * dilation[2]] * 
            weight[c // channel_per_group * in_channel_per_group + rc, c % channel_per_group, k_d - rd - 1, k_h - rh - 1, k_w - rw - 1]),
            axis=[rc, rd, rw, rh])
    )
    if bias is not None:
        output = tvm.compute(
            (batch_size, output_channel, out_d, out_h, out_w), 
            lambda b, c, d, h, w: output[b, c, d, h, w] + bias[c]
            )
    return output
    

def im2col_nchw_naive(inputs, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    """Image to column NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.tensor.Tensor
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
    tvm.tensor.Tensor
        shape [groups, height, width]
    -----------------------------
    """
    # TODO only support tvm.expr.IntImm type
    assert_print(all([isinstance(x, tvm.expr.IntImm) for x in kernel_size]))
    batch_size, channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = kernel_size
    out_channel_per_group = out_channel // groups
    assert_print((channel_per_group * groups).value == channel.value)
    assert_print((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.expr.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.expr.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)
    assert_print(isinstance(groups, (int, tvm.expr.IntImm)))
    
    out_height = k_h * k_w * channel_per_group
    P = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    Q = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    output_width = batch_size * P * Q

    padded = zero_pad2d(inputs,padding=padding)
    output = tvm.compute(
        (groups, out_height, output_width),
        lambda g, h, w: padded[w // (P * Q), 
                            g * channel_per_group + h // (k_h * k_w), 
                            w % (P * Q) // Q * stride[0] + h % (k_h * k_w) // k_w * dilation[0],
                            w % Q * stride[1] + h % k_w * dilation[1]]
        )
    return output


def col2img_nchw(inputs, P, Q):
    """Column to image NCHW layout

    Args:
    -----------------------------
    inputs: tvm.tensor.Tensor
        shape [batch, channel, height, width]
    P     : int or tvm.expr.IntImm
        height of output image
    Q     : int or tvm.expr.IntImm
        width of output image
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, channel, P, Q]
    -----------------------------
    """
    groups, out_channel_per_group, width = inputs.shape
    batch_size = width // (P * Q)
    output = tvm.compute(
        (batch_size, out_channel_per_group * groups, P, Q),
        lambda b, c, h, w: inputs[c // out_channel_per_group, c % out_channel_per_group, b * (P * Q) + h * Q + w]
    )
    return output


def gemm_conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Gemm based convolution 2D NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : tvm.tensor.Tensor
        shape [out_channel, channel_per_group, kernel_height, kernel_width]
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
        shape [batch, out_channel, out_height, out_width]
    -----------------------------
    """
    out_channel, _, k_h, k_w = weight.shape
    out_channel_per_group = out_channel // groups

    _, _, in_h, in_w = inputs.shape
    assert_print((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.expr.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.expr.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.expr.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)
    assert_print(isinstance(groups, (int, tvm.expr.IntImm)))
    
    P = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    Q = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

    col = im2col_nchw_naive(inputs, weight.shape, stride, padding, dilation, groups)
    k = tvm.reduce_axis((0, col.shape[1]))
    gemm = tvm.compute(
        (groups, out_channel_per_group, col.shape[2]),
        lambda g, h, w: tvm.sum(col[g, k, w] * 
                                weight[g * out_channel_per_group + h, 
                                    k // (k_h * k_w), 
                                    k % (k_h * k_w) // k_w, 
                                    k % k_w],
                                axis=k
                                )
                    )
    if bias is not None:
        gemm = tvm.compute(
            (groups, out_channel_per_group, col.shape[2]),
            lambda g, h, w: gemm[g, h, w] + bias[g * out_channel_per_group + h])
    img = col2img_nchw(gemm, P, Q)
    return img


def gemv(A, vector, transposeA=False):
    """Matrix multiplies vector

    Args:
    -----------------------------
    A: tvm.tensor.Tensor
        shape [height, width]
    vector: tvm.tensor.Tensor
        shape [width]
    transposeA: (optional:False) bool
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [height]
    -----------------------------
    """
    k = tvm.reduce_axis((0, vector.shape[0]))
    if transposeA:
        assert_print(A.shape[0].value == vector.shape[0].value)
        return tvm.compute((A.shape[1],), lambda i: tvm.sum(A[k, i] * vector[k], axis=k))
    else:
        assert_print(A.shape[1].value == vector.shape[0].value)
        return tvm.compute((A.shape[0],), lambda i: tvm.sum(A[i, k] * vector[k], axis=k))


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
        assert_print(A.shape[0].value == B.shape[1].value)
        return tvm.compute((A.shape[1], B.shape[0]), lambda i, j: tvm.sum(A[k, i] * B[j, k], axis=k))
    elif transposeA and not transposeB:
        k = tvm.reduce_axis((0, B.shape[0]))
        assert_print(A.shape[0].value == B.shape[0].value)
        return tvm.compute((A.shape[1], B.shape[1]), lambda i, j: tvm.sum(A[k, i] * B[k, j], axis=k))
    elif not transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[1].value == B.shape[1].value)
        return tvm.compute((A.shape[0], B.shape[0]), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k))
    else:
        k = tvm.reduce_axis((0, B.shape[0]))
        assert_print(A.shape[1].value == B.shape[0].value)
        return tvm.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k))


def batch_gemm(A, B, transposeA=False, transposeB=False):
    """Batched matrix multiplies matrix

    Args:
    -----------------------------
    A: tvm.tensor.Tensor
        shape [batch, height, width]
    B: tvm.tensor.Tensor
        shape [batch, width, length]
    transposeA: (optional:False) bool
    transposeB: (optional:False) bool
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, height, length]
    -----------------------------
    """
    assert_print(A.shape[0].value == B.shape[0].value)
    if transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[2]))
        assert_print(A.shape[1].value == B.shape[2].value)
        return tvm.compute(
            (A.shape[0], A.shape[2], B.shape[1]), 
            lambda b, i, j: tvm.sum(A[b, k, i] * B[b, j, k], axis=k)
            )
    elif transposeA and not transposeB:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[1].value == B.shape[1].value)
        return tvm.compute(
            (A.shape[0], A.shape[2], B.shape[2]), 
            lambda b, i, j: tvm.sum(A[b, k, i] * B[b, k, j], axis=k)
            )
    elif not transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[2]))
        assert_print(A.shape[2].value == B.shape[2].value)
        return tvm.compute(
            (A.shape[0], A.shape[1], B.shape[1]), 
            lambda b, i, j: tvm.sum(A[b, i, k] * B[b, j, k], axis=k)
            )
    else:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[2].value == B.shape[1].value)
        return tvm.compute(
            (A.shape[0], A.shape[1], B.shape[2]), 
            lambda b, i, j: tvm.sum(A[b, i, k] * B[b, k, j], axis=k)
            )


def linear(inputs, weight, bias=None):
    """Linear function

    Args:
    -----------------------------
    inputs: tvm.tensor.Tensor
        shape [batch, ..., in_feature]
    weight: tvm.tensor.Tensor
        shape [out_feature, in_feature]
    bias  : tvm.tensor.Tensor
        shape [out_feature]
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, ..., out_feature]
    -----------------------------
    """
    assert_print(inputs.shape[-1].value == weight.shape[1].value)
    k = tvm.reduce_axis((0, inputs.shape[-1]))

    def _inner(*args):
        return tvm.sum(inputs[(*args[:-1], k)] * weight[args[-1], k], axis=k)
    
    output = tvm.compute((*inputs.shape[:-1], weight.shape[0]), _inner)
    if bias is not None:
        assert_print(bias.shape[0].value == weight.shape[0].value)
        def _add(*args):
            return output[args] + bias[args[-1]]
        output = tvm.compute(output.shape, _add)
    return output


def bilinear(inputs1, inputs2, weight, bias=None):
    """Bilinear function

    Args:
    -----------------------------
    inputs1: tvm.tensor.Tensor
        shape [batch, ..., in_feature1]
    inputs2: tvm.tensor.Tensor
        shape [batch, ..., in_feature2]
    weight: tvm.tensor.Tensor
        shape [out_feature, in_feature1, in_feature2]
    bias  : tvm.tensor.Tensor
        shape [out_feature]
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, ..., out_feature]
    -----------------------------
    """
    assert_print(inputs1.shape[-1].value == weight.shape[-2].value)
    assert_print(inputs2.shape[-1].value == weight.shape[-1].value)
    k1 = tvm.reduce_axis((0, inputs1.shape[-1]))
    k2 = tvm.reduce_axis((0, inputs2.shape[-1]))
    for i in range(len(inputs1.shape) - 1):
        assert_print(inputs1.shape[i].value == inputs2.shape[i].value)

    def _inner(*args):
        return tvm.sum(
            inputs1[(*args[:-1], k1)] * weight[args[-1], k1, k2] * inputs2[(*args[:-1], k2)], 
            axis=[k1, k2]
            )
    
    output = tvm.compute((*inputs1.shape[:-1], weight.shape[0]), _inner)
    if bias is not None:
        assert_print(bias.shape[0].value == weight.shape[0].value)
        def _add(*args):
            return output[args] + bias[args[-1]]
        output = tvm.compute(output.shape, _add)
    return output


def MTTKRP3d(A, B, C):
    """Dense MTTKRP 3D

    Args:
    -----------------------------
    A: tvm.tensor.Tensor
        shape [out_height, h1, h2]
    B: tvm.tensor.Tensor
        shape [h1, out_width]
    C: tvm.tensor.Tensor
        shape [h2, out_width]
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [out_height, out_width]
    -----------------------------
    """
    assert_print(A.shape[1].value == B.shape[0].value)
    assert_print(A.shape[2].value == C.shape[0].value)
    assert B.shape[1].value == C.shape[1].value
    k = tvm.reduce_axis((0, B.shape[0]))
    l = tvm.reduce_axis((0, C.shape[0]))

    return tvm.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.sum(A[i, k, l] * B[k, j] * C[l, j], axis=[k, l]))


def pointwise_multiply(A, B):
    """Pointwise multiply

    Args:
    -----------------------------
    A: tvm.tensor.Tensor
        shape [...]
    B: tvm.tensor.Tensor
        shape same as A
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape same as A
    -----------------------------
    """
    assert_print(len(A.shape) == len(B.shape))
    for i in range(len(A.shape)):
        assert_print(A.shape[i].value == B.shape[i].value)
    
    def _mul(*args):
        return A[args] * B[args]

    return tvm.compute(A.shape, _mul)


def mean(inputs, dim=0):
    """Mean

    Args:
    -----------------------------
    A: tvm.tensor.Tensor
        shape [...]
    dim: (optional:0) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [...]
    -----------------------------
    """
    assert_print(dim >=0 and dim < len(inputs.shape))
    output_shape = []
    for i in range(len(inputs.shape)):
        if i == dim:
            continue
        output_shape.append(inputs.shape[i])
    
    k = tvm.reduce_axis((0, inputs.shape[dim]))

    def _inner(*args):
        return tvm.sum(inputs[(*args[:dim], k, *args[dim:])] / inputs.shape[dim], axis=k)
    return tvm.compute(output_shape, _inner)


def variance(inputs, mean_val=None, dim=0):
    """Variance

    Args:
    -----------------------------
    A: tvm.tensor.Tensor
        shape [...]
    dim: (optional:0) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
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
    
    k = tvm.reduce_axis((0, inputs.shape[dim]))
    mean_val = mean_val if mean_val is not None else mean(inputs, dim)

    def _inner(*args):
        return tvm.sum((inputs[(*args[:dim], k, *args[dim:])] - mean_val[args]) * 
                    (inputs[(*args[:dim], k, *args[dim:])] - mean_val[args]) / (inputs.shape[dim] - 1), axis=k)
    return tvm.compute(output_shape, _inner)

def batch_normalization2d(inputs, epsilon=1e-5):
    mean_val = mean(inputs, dim=0)
    var_val = variance(inputs, mean_val=mean_val, dim=0)
    return tvm.compute(inputs.shape, lambda i, j: (inputs[i, j] - mean_val[j]) / tvm.sqrt(var_val[j] + epsilon))

def LSTMCell(inputs, hs, cs, weights, bias=None):
    assert inputs.shape[0].value == hs.shape[0].value
    assert hs.shape[0].value == cs.shape[0].value
    assert weights.shape[0].value == 4
    assert weights.shape[2].value == inputs.shape[1].value + hs.shape[1].value
    k1 = tvm.reduce_axis((0, inputs.shape[1]))
    k2 = tvm.reduce_axis((0, hs.shape[1]))
    A = tvm.compute(
        (inputs.shape[0], weights.shape[0], weights.shape[1]), 
        lambda b, i, j: tvm.sum(inputs[b, k1] * weights[i, j, k1], axis=k1))
    B = tvm.compute(
        (hs.shape[0], weights.shape[0], weights.shape[1]), 
        lambda b, i, j: tvm.sum(hs[b, k2] * weights[i, j, k2 + inputs.shape[1]], axis=k2))
    if bias is not None:
        C = tvm.compute(
            (inputs.shape[0], weights.shape[0], weights.shape[1]), 
            lambda b, i, j: A[b, i, j] + B[b, i, j] + bias[b, i, j])
    else:
        C = tvm.compute(
            (inputs.shape[0], weights.shape[0], weights.shape[1]), 
            lambda b, i, j: A[b, i, j] + B[b, i, j])
    next_cs = tvm.compute(
        (cs.shape[0], weights.shape[1]), 
        lambda b, i: tvm.sigmoid(C[b, 1, i]) * cs[b, i] + tvm.sigmoid(C[b, 0, i]) * tvm.tanh(C[b, 3, i]))
    next_hs = tvm.compute(
        (hs.shape[0], weights.shape[1]), 
        lambda b, i: tvm.sigmoid(C[b, 2, i]) * tvm.tanh(next_cs[b, i]))
    return next_hs, next_cs

def block_circulant_matrix(Input, factor):
    ROW, COL = Input.shape
    FFT = factor

    k = tvm.reduce_axis((0, FFT))
    Compress = tvm.compute(
        (ROW // FFT, COL),
        lambda i, j: (
            tvm.sum(
                Input[i * FFT + k, (j // FFT) * FFT + (j % FFT + k) % FFT] / FFT,
                axis=k
            )
        )
    )

    Output = tvm.compute(
        (ROW, COL),
        lambda i, j: (
            Compress[i // FFT, (j // FFT) * FFT + ((j % FFT) + FFT - (i % FFT)) % FFT]
        )
    )

    return Output