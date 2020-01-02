from __future__ import absolute_import


import numpy as np


def conv2d_nchwc(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 2d NCHWc layout

    Args:
    -----------------------------
    inputs  : np.ndarray
        shape [batch, channel // vlen, height, width, vlen]
    weight  : np.ndarray
        shape [out_channel // vlen, channel // vlen // groups, kernel_height, kernel_width, vlen(i), vlen(o)]
    bias    : (optional:None) np.ndarray
        shape [out_channel // vlen, vlen]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    np.ndarray
        shape [batch, out_channel // vlen, output_height, output_width, vlen]
    -----------------------------
    """
    batch_size, in_channel_chunk, in_h, in_w, in_channel_block = inputs.shape
    out_channel_chunk, channel_per_group_chunk, k_h, k_w, _in_channel_block, out_channel_block = weight.shape
    assert ((channel_per_group_chunk * groups) == in_channel_chunk)
    assert _in_channel_block == in_channel_block
    assert in_channel_block == out_channel_block
    out_channel_per_group = out_channel_chunk // groups
    assert ((out_channel_per_group * groups) == out_channel_chunk)

    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    assert (isinstance(stride, tuple) and len(stride) == 2)
    assert (isinstance(padding, tuple) and len(padding) == 2)
    assert (isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

    output = np.zeros((batch_size, out_channel_chunk, out_h, out_w, out_channel_block), dtype=inputs.dtype)
    for b in range(batch_size):
        for c_c in range(out_channel_chunk):
            for h in range(out_h):
                for w in range(out_w):
                    for c_b in range(out_channel_block):
                        for rc_chunk in range(channel_per_group_chunk):
                            for rc_block in range(in_channel_block):
                                for rh in range(k_h):
                                    for rw in range(k_w):
                                        h_index = h * stride[0] + rh * dilation[0] - padding[0]
                                        w_index = w * stride[1] + rw * dilation[1] - padding[1]
                                        if 0 <= h_index < in_h and 0 <= w_index < in_w:
                                            output[b, c_c, h, w, c_b] += inputs[b, 
                                                    c_c // out_channel_per_group * channel_per_group_chunk + rc_chunk,
                                                    h_index, 
                                                    w_index, 
                                                    rc_block] * weight[c_c, rc_chunk, rh, rw, rc_block, c_b]
    
    if bias is not None:
        for b in range(batch_size):
            for c_c in range(out_channel_chunk):
                for h in range(out_h):
                    for w in range(out_w):
                        for c_b in range(out_channel_block):
                            output[b, c_c, h, w, c_b] += bias[c_c, c_b]
    
    return output