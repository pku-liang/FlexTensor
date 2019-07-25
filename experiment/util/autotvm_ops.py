from topi.util import equal_const_int, get_const_tuple
from topi import nn, generic
import tvm


def conv_transpose1d(cfg, Input, Filter, strides, padding, out_dtype='float32'):
    batch, in_c, in_l = get_const_tuple(Input.shape)
    _, out_c, filter_l = get_const_tuple(Filter.shape)
    stride_l, = strides

    cfg.stride = strides

    fpad_left, fpad_right = padding[0], padding[0]
    bpad_left = filter_l - 1 - fpad_left
    bpad_right = filter_l - 1 - fpad_right

    FirstPad = nn.pad(
        Input,
        [0, 0, (bpad_left + stride_l - 1) // stride_l],
        [0, 0, (bpad_right + stride_l - 1) // stride_l],
        name='FirstPad'
    )

    border_l = (stride_l - bpad_left % stride_l) % stride_l

    data = FirstPad
    strides = [1, 1, stride_l]
    n = len(data.shape)

    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not equal_const_int(strides[i], 1):
                index_tuple.append(indices[i] // strides[i])
                not_zero.append((indices[i] % strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm.all(*not_zero)
            return tvm.if_then_else(not_zero, data(*index_tuple), tvm.const(0.0, data.dtype))
        return data(*index_tuple)

    out_l = (in_l - 1) * stride_l - fpad_left - fpad_right + filter_l
    dc = tvm.reduce_axis((0, in_c), name='dc')
    dl = tvm.reduce_axis((0, filter_l), name='dl')

    Output = tvm.compute(
        (batch, out_c, out_l),
        lambda b, c, l: (
            tvm.sum(
                _dilate(b, dc, l + dl + border_l).astype(out_dtype) *
                Filter[dc, c, filter_l - 1 - dl].astype(out_dtype),
                axis=[dc, dl]
            )
        )
    )
    return Output


def conv_transpose2d(cfg, Input, Filter, strides, padding, out_dtype='float32'):
    batch, in_c, in_h, in_w = get_const_tuple(Input.shape)
    _, out_c, filter_h, filter_w = get_const_tuple(Filter.shape)
    stride_h, stride_w = strides

    # attach stride info to config, this is used in schedule space definition
    cfg.stride = strides

    # padding stage
    fpad_top, fpad_left, fpad_bottom, fpad_right = nn.get_pad_tuple(
        padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right

    # padding stage
    FirstPad = nn.pad(Input,
                      [0, 0, (bpad_top + stride_h - 1) // stride_h,
                       (bpad_left + stride_w - 1) // stride_w],
                      [0, 0, (bpad_bottom + stride_h - 1) // stride_h,
                       (bpad_right + stride_w - 1) // stride_w], name='FirstPad')

    # remove extra padding introduced by dilatation
    border_h = (stride_h - bpad_top % stride_h) % stride_h
    border_w = (stride_w - bpad_left % stride_w) % stride_w

    # dilation stage
    data = FirstPad
    strides = [1, 1, stride_h, stride_w]
    n = len(data.shape)

    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not equal_const_int(strides[i], 1):
                index_tuple.append(indices[i] // strides[i])
                not_zero.append((indices[i] % strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm.all(*not_zero)
            return tvm.if_then_else(not_zero, data(*index_tuple), tvm.const(0.0, data.dtype))
        return data(*index_tuple)

    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    dc = tvm.reduce_axis((0, in_c), name='dc')
    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')

    Output = tvm.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: tvm.sum(
            _dilate(b, dc, h + dh + border_h, w + dw + border_w).astype(out_dtype) *
            Filter[dc, c, filter_h - 1 - dh,
                   filter_w - 1 - dw].astype(out_dtype),
            axis=[dc, dh, dw]), tag="conv2d_transpose_nchw")

    return Output


def conv_transpose3d(cfg, Input, Filter, strides, padding, out_dtype='float32'):
    batch, in_c, in_d, in_h, in_w = get_const_tuple(Input.shape)
    _, out_c, filter_d, filter_h, filter_w = get_const_tuple(Filter.shape)
    stride_d, stride_h, stride_w = strides

    cfg.stride = strides

    fpad_d, fpad_h, fpad_w = padding

    bpad_d = filter_d - 1 - fpad_d
    bpad_h = filter_h - 1 - fpad_h
    bpad_w = filter_w - 1 - fpad_w

    FirstPad = nn.pad(
        Input,
        [0, 0, (bpad_d + stride_d - 1) // stride_d, (bpad_h +
                                                     stride_h - 1) // stride_h, (bpad_w + stride_w - 1) // stride_w],
        [0, 0, (bpad_d + stride_d - 1) // stride_d, (bpad_h +
                                                     stride_h - 1) // stride_h, (bpad_w + stride_w - 1) // stride_w],
        name="FirstPad"
    )

    border_d = (stride_d - bpad_d % stride_d) % stride_d
    border_h = (stride_h - bpad_h % stride_h) % stride_h
    border_w = (stride_w - bpad_w % stride_w) % stride_w

    data = FirstPad
    strides = [1, 1, stride_d, stride_h, stride_w]
    n = len(data.shape)

    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not equal_const_int(strides[i], 1):
                index_tuple.append(indices[i] // strides[i])
                not_zero.append((indices[i] % strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm.all(*not_zero)
            return tvm.if_then_else(not_zero, data(*index_tuple), tvm.const(0.0, data.dtype))
        return data(*index_tuple)

    out_d = (in_d - 1) * stride_d - fpad_d - fpad_d + filter_d
    out_h = (in_h - 1) * stride_h - fpad_h - fpad_h + filter_h
    out_w = (in_w - 1) * stride_w - fpad_w - fpad_w + filter_w

    dc = tvm.reduce_axis((0, in_c), name='dc')
    dd = tvm.reduce_axis((0, filter_d), name='dd')
    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')

    Output = tvm.compute(
        (batch, out_c, out_d, out_h, out_w),
        lambda b, c, d, h, w: (
            tvm.sum(
                _dilate(b, dc, d + dd + border_d, h + dh + border_h, w + dw + border_w).astype(out_dtype) *
                Filter[dc, c, filter_d - 1 - dd, filter_h - 1 -
                       dh, filter_w - 1 - dw].astype(out_dtype),
                axis=[dc, dd, dh, dw]
            )
        )
    )

    return Output
