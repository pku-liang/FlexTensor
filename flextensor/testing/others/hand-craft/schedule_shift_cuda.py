import tvm
import math
import torch
import numpy as np
from flextensor.nn import ShiftConv2d_nhwc

shift_conv2d_shape = [
    # ShiftNet(https://arxiv.org/abs/1801.09392) with input size: 256*256
    (1, 128, 128, 64, 3, 1), 
    # (1, 128, 128, 64, 3, 1), 
    (1, 64, 64, 128, 5, 1), 
    (1, 32, 32, 256, 3, 1), 
    (1, 16, 16, 512, 3, 1)
]

DEV_ID = 0


def schedule_shift_1_cuda(s, Img, KernelIndex, Output):
    Pad = s[Output].op.input_tensors[1]
    s[Pad].compute_inline()

    write_cache = s.cache_write(Output, "local")
    share_kernel = s.cache_read(s[write_cache].op.input_tensors[0], "shared", [write_cache])
    share_pad = s.cache_read(s[write_cache].op.input_tensors[1], "shared", [write_cache])

    n, h, w, c = s[Output].op.axis
    # 1, 126, 126, 64
    n_factors = [1, 1, 1, 1]
    h_factors = [18, 1, 7, 1]
    w_factors = [9, 7, 2, 1]
    c_factors = [2, 1, 32, 1]

    bn, ni = s[Output].split(n, nparts=n_factors[0])
    vn, ni = s[Output].split(ni, nparts=n_factors[1])
    nm, ni = s[Output].split(ni, nparts=n_factors[2])

    bh, hi = s[Output].split(h, nparts=h_factors[0])
    vh, hi = s[Output].split(hi, nparts=h_factors[1])
    hm, hi = s[Output].split(hi, nparts=h_factors[2])

    bw, wi = s[Output].split(w, nparts=w_factors[0])
    vw, wi = s[Output].split(wi, nparts=w_factors[1])
    wm, wi = s[Output].split(wi, nparts=w_factors[2])

    bc, ci = s[Output].split(c, nparts=c_factors[0])
    vc, ci = s[Output].split(ci, nparts=c_factors[1])
    cm, ci = s[Output].split(ci, nparts=c_factors[2])

    s[Output].reorder(bn, bh, bw, bc, vn, vh, vw, vc, nm, hm, wm, cm, ni, hi, wi, ci)

    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    vx = tvm.te.thread_axis("vthread")
    vy = tvm.te.thread_axis("vthread")
    vz = tvm.te.thread_axis("vthread")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")

    s[Output].bind(bh, bz)
    s[Output].bind(bw, by)
    s[Output].bind(bc, bx)
    s[Output].bind(vh, vz)
    s[Output].bind(vw, vy)
    s[Output].bind(vc, vx)
    s[Output].bind(hm, tz)
    s[Output].bind(wm, ty)
    s[Output].bind(cm, tx)
    
    s[write_cache].compute_at(s[Output], cm)
    a, b, c, d = s[write_cache].op.axis
    s[share_pad].compute_inline()
    s[share_kernel].compute_at(s[write_cache], a)

    # a, b, c, d = s[share_pad].op.axis
    # pad_z, b = s[share_pad].split(b, nparts=h_factors[2])
    # pad_y, c = s[share_pad].split(c, nparts=w_factors[2])
    # pad_x, d = s[share_pad].split(d, nparts=c_factors[2])
    # s[share_pad].bind(pad_z, tz)
    # s[share_pad].bind(pad_y, ty)
    # s[share_pad].bind(pad_x, tx)

    fused = s[share_kernel].op.axis[0]
    # pad_z, fused = s[share_kernel].split(fused, nparts=h_factors[2])
    pad_y, fused = s[share_kernel].split(fused, nparts=w_factors[2])
    pad_x, fused = s[share_kernel].split(fused, nparts=c_factors[2])
    # s[share_kernel].bind(pad_z, tz)
    s[share_kernel].bind(pad_y, ty)
    s[share_kernel].bind(pad_x, tx)

    s[Output].pragma(bn, 'auto_unroll_max_step', 200)
    s[Output].pragma(bn, 'unroll_explicit', 1)



def schedule_shift_2_cuda(s, Img, KernelIndex, Output):
    Pad = s[Output].op.input_tensors[1]
    s[Pad].compute_inline()

    write_cache = s.cache_write(Output, "local")
    share_kernel = s.cache_read(s[write_cache].op.input_tensors[0], "shared", [write_cache])
    share_pad = s.cache_read(s[write_cache].op.input_tensors[1], "shared", [write_cache])

    n, h, w, c = s[Output].op.axis
    # 1, 60, 60, 128
    n_factors = [1, 1, 1, 1]
    h_factors = [12, 1, 5, 1]
    w_factors = [15, 1, 4, 1]
    c_factors = [4, 1, 32, 1]

    bn, ni = s[Output].split(n, nparts=n_factors[0])
    vn, ni = s[Output].split(ni, nparts=n_factors[1])
    nm, ni = s[Output].split(ni, nparts=n_factors[2])

    bh, hi = s[Output].split(h, nparts=h_factors[0])
    vh, hi = s[Output].split(hi, nparts=h_factors[1])
    hm, hi = s[Output].split(hi, nparts=h_factors[2])

    bw, wi = s[Output].split(w, nparts=w_factors[0])
    vw, wi = s[Output].split(wi, nparts=w_factors[1])
    wm, wi = s[Output].split(wi, nparts=w_factors[2])

    bc, ci = s[Output].split(c, nparts=c_factors[0])
    vc, ci = s[Output].split(ci, nparts=c_factors[1])
    cm, ci = s[Output].split(ci, nparts=c_factors[2])

    s[Output].reorder(bn, bh, bw, bc, vn, vh, vw, vc, nm, hm, wm, cm, ni, hi, wi, ci)

    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    vx = tvm.te.thread_axis("vthread")
    vy = tvm.te.thread_axis("vthread")
    vz = tvm.te.thread_axis("vthread")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")

    s[Output].bind(bh, bz)
    s[Output].bind(bw, by)
    s[Output].bind(bc, bx)
    s[Output].bind(vh, vz)
    s[Output].bind(vw, vy)
    s[Output].bind(vc, vx)
    s[Output].bind(hm, tz)
    s[Output].bind(wm, ty)
    s[Output].bind(cm, tx)
    
    s[write_cache].compute_at(s[Output], cm)
    a, b, c, d = s[write_cache].op.axis
    s[share_pad].compute_inline()
    s[share_kernel].compute_at(s[write_cache], a)

    # a, b, c, d = s[share_pad].op.axis
    # pad_z, b = s[share_pad].split(b, nparts=h_factors[2])
    # pad_y, c = s[share_pad].split(c, nparts=w_factors[2])
    # pad_x, d = s[share_pad].split(d, nparts=c_factors[2])
    # s[share_pad].bind(pad_z, tz)
    # s[share_pad].bind(pad_y, ty)
    # s[share_pad].bind(pad_x, tx)

    fused = s[share_kernel].op.axis[0]
    # pad_z, fused = s[share_kernel].split(fused, nparts=h_factors[2])
    pad_y, fused = s[share_kernel].split(fused, nparts=w_factors[2])
    pad_x, fused = s[share_kernel].split(fused, nparts=c_factors[2])
    # s[share_kernel].bind(pad_z, tz)
    s[share_kernel].bind(pad_y, ty)
    s[share_kernel].bind(pad_x, tx)

    s[Output].pragma(bn, 'auto_unroll_max_step', 200)
    s[Output].pragma(bn, 'unroll_explicit', 1)


def schedule_shift_3_cuda(s, Img, KernelIndex, Output):
    Pad = s[Output].op.input_tensors[1]
    s[Pad].compute_inline()

    write_cache = s.cache_write(Output, "local")
    share_kernel = s.cache_read(s[write_cache].op.input_tensors[0], "shared", [write_cache])
    share_pad = s.cache_read(s[write_cache].op.input_tensors[1], "shared", [write_cache])

    n, h, w, c = s[Output].op.axis
    # 1, 30, 30, 256
    n_factors = [1, 1, 1, 1]
    h_factors = [15, 1, 2, 1]
    w_factors = [15, 1, 2, 1]
    c_factors = [2, 1, 128, 1]

    bn, ni = s[Output].split(n, nparts=n_factors[0])
    vn, ni = s[Output].split(ni, nparts=n_factors[1])
    nm, ni = s[Output].split(ni, nparts=n_factors[2])

    bh, hi = s[Output].split(h, nparts=h_factors[0])
    vh, hi = s[Output].split(hi, nparts=h_factors[1])
    hm, hi = s[Output].split(hi, nparts=h_factors[2])

    bw, wi = s[Output].split(w, nparts=w_factors[0])
    vw, wi = s[Output].split(wi, nparts=w_factors[1])
    wm, wi = s[Output].split(wi, nparts=w_factors[2])

    bc, ci = s[Output].split(c, nparts=c_factors[0])
    vc, ci = s[Output].split(ci, nparts=c_factors[1])
    cm, ci = s[Output].split(ci, nparts=c_factors[2])

    s[Output].reorder(bn, bh, bw, bc, vn, vh, vw, vc, nm, hm, wm, cm, ni, hi, wi, ci)

    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    vx = tvm.te.thread_axis("vthread")
    vy = tvm.te.thread_axis("vthread")
    vz = tvm.te.thread_axis("vthread")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")

    s[Output].bind(bh, bz)
    s[Output].bind(bw, by)
    s[Output].bind(bc, bx)
    s[Output].bind(vh, vz)
    s[Output].bind(vw, vy)
    s[Output].bind(vc, vx)
    s[Output].bind(hm, tz)
    s[Output].bind(wm, ty)
    s[Output].bind(cm, tx)
    
    s[write_cache].compute_at(s[Output], cm)
    a, b, c, d = s[write_cache].op.axis
    s[share_pad].compute_inline()
    s[share_kernel].compute_at(s[write_cache], a)

    # a, b, c, d = s[share_pad].op.axis
    # pad_z, b = s[share_pad].split(b, nparts=h_factors[2])
    # pad_y, c = s[share_pad].split(c, nparts=w_factors[2])
    # pad_x, d = s[share_pad].split(d, nparts=c_factors[2])
    # s[share_pad].bind(pad_z, tz)
    # s[share_pad].bind(pad_y, ty)
    # s[share_pad].bind(pad_x, tx)

    fused = s[share_kernel].op.axis[0]
    # pad_z, fused = s[share_kernel].split(fused, nparts=h_factors[2])
    pad_y, fused = s[share_kernel].split(fused, nparts=w_factors[2])
    pad_x, fused = s[share_kernel].split(fused, nparts=c_factors[2])
    # s[share_kernel].bind(pad_z, tz)
    s[share_kernel].bind(pad_y, ty)
    s[share_kernel].bind(pad_x, tx)

    s[Output].pragma(bn, 'auto_unroll_max_step', 200)
    s[Output].pragma(bn, 'unroll_explicit', 1)


def schedule_shift_4_cuda(s, Img, KernelIndex, Output):
    Pad = s[Output].op.input_tensors[1]
    s[Pad].compute_inline()

    write_cache = s.cache_write(Output, "local")
    share_kernel = s.cache_read(s[write_cache].op.input_tensors[0], "shared", [write_cache])
    share_pad = s.cache_read(s[write_cache].op.input_tensors[1], "shared", [write_cache])

    n, h, w, c = s[Output].op.axis
    # 1, 14, 14, 512
    n_factors = [1, 1, 1, 1]
    h_factors = [7, 1, 2, 1]
    w_factors = [7, 1, 2, 1]
    c_factors = [4, 1, 128, 1]

    bn, ni = s[Output].split(n, nparts=n_factors[0])
    vn, ni = s[Output].split(ni, nparts=n_factors[1])
    nm, ni = s[Output].split(ni, nparts=n_factors[2])

    bh, hi = s[Output].split(h, nparts=h_factors[0])
    vh, hi = s[Output].split(hi, nparts=h_factors[1])
    hm, hi = s[Output].split(hi, nparts=h_factors[2])

    bw, wi = s[Output].split(w, nparts=w_factors[0])
    vw, wi = s[Output].split(wi, nparts=w_factors[1])
    wm, wi = s[Output].split(wi, nparts=w_factors[2])

    bc, ci = s[Output].split(c, nparts=c_factors[0])
    vc, ci = s[Output].split(ci, nparts=c_factors[1])
    cm, ci = s[Output].split(ci, nparts=c_factors[2])

    s[Output].reorder(bn, bh, bw, bc, vn, vh, vw, vc, nm, hm, wm, cm, ni, hi, wi, ci)

    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    vx = tvm.te.thread_axis("vthread")
    vy = tvm.te.thread_axis("vthread")
    vz = tvm.te.thread_axis("vthread")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")

    s[Output].bind(bh, bz)
    s[Output].bind(bw, by)
    s[Output].bind(bc, bx)
    s[Output].bind(vh, vz)
    s[Output].bind(vw, vy)
    s[Output].bind(vc, vx)
    s[Output].bind(hm, tz)
    s[Output].bind(wm, ty)
    s[Output].bind(cm, tx)
    
    s[write_cache].compute_at(s[Output], cm)
    a, b, c, d = s[write_cache].op.axis
    s[share_pad].compute_inline()
    s[share_kernel].compute_at(s[write_cache], a)

    # a, b, c, d = s[share_pad].op.axis
    # pad_z, b = s[share_pad].split(b, nparts=h_factors[2])
    # pad_y, c = s[share_pad].split(c, nparts=w_factors[2])
    # pad_x, d = s[share_pad].split(d, nparts=c_factors[2])
    # s[share_pad].bind(pad_z, tz)
    # s[share_pad].bind(pad_y, ty)
    # s[share_pad].bind(pad_x, tx)

    fused = s[share_kernel].op.axis[0]
    pad_z, fused = s[share_kernel].split(fused, nparts=h_factors[2])
    pad_y, fused = s[share_kernel].split(fused, nparts=w_factors[2])
    pad_x, fused = s[share_kernel].split(fused, nparts=c_factors[2])
    s[share_kernel].bind(pad_z, tz)
    s[share_kernel].bind(pad_y, ty)
    s[share_kernel].bind(pad_x, tx)

    s[Output].pragma(bn, 'auto_unroll_max_step', 200)
    s[Output].pragma(bn, 'unroll_explicit', 1)



def evaluate(shape, schedule_func):
    N, H, W, C, k, dilation = shape
    stride = 1
    Img = tvm.te.placeholder([N, H, W, C], dtype="float32")
    KernelIndex = tvm.te.placeholder([C], dtype="int32")
    Output = ShiftConv2d_nhwc(Img, KernelIndex, k, dilation, stride)

    s = tvm.te.create_schedule(Output.op)
    schedule_func(s, Img, KernelIndex, Output)

    func = tvm.build(s, [Img, KernelIndex, Output], "cuda")
    # print(func.imported_modules[0].get_source())
    Img_torch = torch.rand([N, H, W, C], dtype=torch.float32)
    Kernel_torch = torch.rand([C, k, k], dtype=torch.float32)
    KernelIndex_torch = torch.argmax(Kernel_torch.reshape([C, -1]), dim=1)

    paddings = [math.ceil(((stride - 1) * H - stride + dilation * (k - 1)) / 2), 
                math.ceil(((stride - 1) * W - stride + dilation * (k - 1)) / 2)]
    image_height = H
    image_width = W
    out_height = math.floor((image_height + 2 * paddings[0]- dilation * (k - 1) - 1) / stride + 1)
    out_width = math.floor((image_width + 2 * paddings[1] - dilation * (k - 1) - 1) / stride + 1)
    output_shape = (N, out_height, out_width, C)

    Output_torch = torch.zeros(output_shape, dtype=torch.float32)

    ctx = tvm.context("cuda", DEV_ID)

    Img_tvm = tvm.nd.array(Img_torch.numpy().astype(np.float32), ctx)
    KernelIndex_tvm = tvm.nd.array(KernelIndex_torch.numpy().astype(np.int32), ctx)
    Output_tvm = tvm.nd.array(Output_torch.numpy().astype(np.float32), ctx)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    time_cost = evaluator(Img_tvm, KernelIndex_tvm, Output_tvm).mean * 1e3

    return time_cost



def main():
    print(evaluate(shift_conv2d_shape[0], schedule_shift_1_cuda))
    print(evaluate(shift_conv2d_shape[1], schedule_shift_2_cuda))
    print(evaluate(shift_conv2d_shape[2], schedule_shift_3_cuda))
    print(evaluate(shift_conv2d_shape[3], schedule_shift_4_cuda))


if __name__ == "__main__":
    main()