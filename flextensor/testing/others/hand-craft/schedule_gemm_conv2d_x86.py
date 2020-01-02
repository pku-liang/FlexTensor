"""
High performance schedule for gemm conv2d
Target x86 CPU
====================================
**Author**: `Kaiwen Sheng`
**Machine Specification"": 
    Linux DellT7910 4.15.0-52-generic #56~16.04.1-Ubuntu SMP Thu Jun 6 12:03:31 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
    cpu             : Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
    cpu cores       : 14
    number of cpu   : 56
    MemTotal        : 528300924 kB
    
    gpu             : NVIDIA Corporation GM200GL [Quadro M6000 24GB] (rev a1)
    number of gpu   : 1
"""
import tvm 
from flextensor.measure import _evaluate
from flextensor.nn import SqueezeNetFire8Gemm
from flextensor.utils import test_allclose, to_tuple
import torch
import numpy as np
import time


"""def schedule_yolo_conv_x86(s, outputs, inputs, weight, bias):
    # inline the padding operation
    padded = outputs.op.input_tensors[0]
    
    # create cache
    write_cache = s.cache_write(outputs, "local")
    # tunable parameters
    b_factors = [1, 1, 1]
    k_factors = [64, 1, 1]
    p_factors = [7, 1, 64]
    q_factors = [14, 1, 32]
    rc_factors = [1, 1, 3]         # outer-->inner
    ry_factors = [1, 1, 7]
    rx_factors = [1, 1, 7]
    # split the spatial axes
    b, k, p, q = s[outputs].op.axis
    bo, bi = s[outputs].split(b, nparts=b_factors[0])
    ko, ki = s[outputs].split(k, nparts=k_factors[0])
    po, pi = s[outputs].split(p, nparts=p_factors[0])
    qo, qi = s[outputs].split(q, nparts=q_factors[0])
    vbo, bi = s[outputs].split(bi, nparts=b_factors[1])
    vko, ki = s[outputs].split(ki, nparts=k_factors[1])
    vpo, pi = s[outputs].split(pi, nparts=p_factors[1])
    vqo, qi = s[outputs].split(qi, nparts=q_factors[1])
    # reorder
    s[outputs].reorder(bo, ko, po, qo, vbo, vko, vpo, vqo, bi, ki, pi, qi)
    s[bias].compute_at(s[outputs], ko)
    print(outputs.op.axis)
    print(s[bias])
    # fuse
    outer = s[outputs].fuse(bo, ko, po, qo)
    middle = s[outputs].fuse(vbo, vko, vpo, vqo)
    # inner = s[outputs].fuse(bi, ki, pi, qi)
    # vectorize
    # s[outputs].vectorize(inner)
    # parallel
    s[outputs].parallel(outer)
    # compute at write cache
    s[write_cache].compute_at(s[outputs], middle)
    # split reduce axes
    wb, wk, wp, wq = s[write_cache].op.axis
    print(s[write_cache].op.reduce_axis)
    gc, ry, rx = s[write_cache].op.reduce_axis
    rco, rci = s[write_cache].split(gc, nparts=rc_factors[0])
    rcm, rci = s[write_cache].split(rci, nparts=rc_factors[1])
    rxo, rxi = s[write_cache].split(rx, nparts=rx_factors[0])
    rxm, rxi = s[write_cache].split(rxi, nparts=rx_factors[1])
    ryo, ryi = s[write_cache].split(ry, nparts=ry_factors[0])
    rym, ryi = s[write_cache].split(ryi, nparts=ry_factors[1])
    # reorder
    s[write_cache].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, wb, wk, wp, wq)
    
    s[outputs].pragma(outer, 'auto_unroll_max_step', 1500)
    # s[write_cache].vectorize(s[write_cache].op.axis[-1])
    s[padded].compute_inline()"""

def schedule_yolo_conv_x86(s, outputs, inputs, weight, bias):
    """[
        stage(inputes, 0x555fc17293e0), 
        stage(Padding, 0x555fc17295d0), 
        stage(Img2Col, 0x555fc1729730), 
        stage(weights, 0x555fc1729950), 
        stage(ComputeGemm, 0x555fc1729b50), 
        stage(bias, 0x555fc1729d90), 
        stage(ComputeBias, 0x555fc1729fc0), 
        stage(Col2Img, 0x555fc172a150)]
    """
    out_channel, in_channel, kh, kw = weight.shape

    Padding = s.stages[1].op
    Img2Col = s.stages[2].op
    ComputeGemm = s.stages[4].op
    ComputeBias = s.stages[6].op
    Col2Img = s.stages[-1].op

    b, c, h, w = s[Padding].op.axis
    batch_size = b.dom.extent
    print("Batch: ", batch_size)
    
    gv = s[ComputeGemm].op.reduce_axis[0]
    """gc, gt = s[ComputeGemm].split(gv, nparts=c.dom.extent.value)
    gh, gw = s[ComputeGemm].split(gt, nparts=h.dom.extent.value)"""

    ig, iv, ib = s[Img2Col].op.axis
    """ic, it = s[Img2Col].split(iv, nparts=c.dom.extent.value)
    ih, iw = s[Img2Col].split(it, nparts=h.dom.extent.value)"""

    # inline padding
    s[Padding].compute_at(s[Img2Col], ib)
    s[Padding].compute_inline()

    # split and reorder Img2Col

    # gemm
    # cache write
    bias_tensor = outputs.op.input_tensors[0]
    gemm_tensor = bias_tensor.op.input_tensors[0]
    write_cache = s.cache_write(gemm_tensor, 'global')

    gemm_cache_g, gemm_cache_h, gemm_cache_w = s[write_cache].op.axis
    gemm_cache_rv = s[write_cache].op.reduce_axis[0]
    h_factor = [8]
    w_factor = [27]
    rv_factor = [8 * kh.value, 8 * kw.value]

    gemm_cache_wo, gemm_cache_wi = s[write_cache].split(gemm_cache_w, nparts=w_factor[0])
    gemm_cache_wio, gemm_cache_wii = s[write_cache].split(gemm_cache_wi, nparts=batch_size)

    gemm_cache_ho, gemm_cache_hi = s[write_cache].split(gemm_cache_h, nparts=h_factor[0])

    gemm_cache_rvo, gemm_cache_rvi = s[write_cache].split(gemm_cache_rv, nparts=rv_factor[0])
    gemm_cache_rvmo, gemm_cache_rvmi = s[write_cache].split(gemm_cache_rvi, nparts=rv_factor[1])
    """gemm_cache_rvno, gemm_cache_rvni = s[write_cache].split(gemm_cache_rvmi, nparts=rv_factor[2])
    gemm_cache_rvlo, gemm_cache_rvli = s[write_cache].split(gemm_cache_rvni, nparts=rv_factor[3])"""

    # s[write_cache].reorder(gemm_cache_g, gemm_cache_ho, gemm_cache_rvmo, gemm_cache_hi, gemm_cache_rvo, gemm_cache_wo, gemm_cache_wi) # good one
    # outer = s[write_cache].fuse(gemm_cache_g, gemm_cache_ho, gemm_cache_rvmo)
    
    s[write_cache].reorder(gemm_cache_g, gemm_cache_wio, gemm_cache_ho, gemm_cache_hi, gemm_cache_rvmo, gemm_cache_rvmi, gemm_cache_wo, gemm_cache_rvo, gemm_cache_wii) # good one
    cache_outer = s[write_cache].fuse(gemm_cache_g, gemm_cache_wio, gemm_cache_ho, gemm_cache_hi)

    # The same split on gemm operato
    # r# cache write
    print(s[gemm_tensor].op.axis)
    gemm_g, gemm_h, gemm_w = s[gemm_tensor].op.axis
    h_factor = [8]
    w_factor = [27]
    rv_factor = [8, 8]

    gemm_wo, gemm_wi = s[gemm_tensor].split(gemm_w, nparts=w_factor[0])

    gemm_ho, gemm_hi = s[gemm_tensor].split(gemm_h, nparts=h_factor[0])

    """gemm_rvo, gemm_rvi = s[gemm_tensor].split(gemm_rv, nparts=rv_factor[0])
    gemm_rvmo, gemm_rvmi = s[gemm_tensor].split(gemm_rvi, nparts=rv_factor[1])
    gemm_rvno, gemm_rvni = s[gemm_tensor].split(gemm_rvmi, nparts=rv_factor[2])
    gemm_rvlo, gemm_rvli = s[gemm_tensor].split(gemm_rvni, nparts=rv_factor[3])"""

    # s[gemm_tensor].reorder(gemm_g, gemm_ho, gemm_rvmo, gemm_hi, gemm_rvo, gemm_wo, gemm_wi) # good one
    # outer = s[gemm_tensor].fuse(gemm_g, gemm_ho, gemm_rvmo)
    
    s[gemm_tensor].reorder(gemm_g, gemm_ho, gemm_hi, gemm_wo, gemm_wi) # good one
    outer = s[gemm_tensor].fuse(gemm_g, gemm_ho, gemm_hi)
    
    # sc
    # s[gemm_tensor].reorder(gemm_g, gemm_go, gemm_rvo, gemm_rvmo, gemm_gi, gemm_w, gemm_rvno, gemm_rvlo, gemm_rvli)
    # outer = s[gemm_tensor].fuse(gemm_g, gemm_go, gemm_rvo)
    
    # Yolo conv24 use 148.2433819 ms
    # Pytorch on cpu use: 1.0495901107788086ms
    # s[gemm_tensor].reorder(gemm_g, gemm_go, gemm_rvno, gemm_rvmo, gemm_gi, gemm_rvo, gemm_rvlo, gemm_rvli, gemm_w)
    # outer = s[gemm_tensor].fuse(gemm_g, gemm_go, gemm_rvno)

    s[write_cache].parallel(cache_outer)
    # s[write_cache].compute_at(s[gemm_tensor], outer)

    # inline bias
    cb, cc, ch, cw = s[Col2Img].op.axis
    s[ComputeBias].compute_at(s[Col2Img], ch)
    s[ComputeBias].compute_inline()

    s[Col2Img].parallel(cc)

    return s


def try_yolo_conv(batch_size=2, number=100):
    # get the compute
    yolo_conv = SqueezeNetFire8Gemm()
    input_shape = yolo_conv.get_intput_shape()
    inputs = tvm.placeholder((batch_size, *input_shape), dtype="float32", name='inputs')
    weight = yolo_conv.get_weight()
    outputs = yolo_conv(inputs)
    bias = yolo_conv.get_bias()
    
    s = tvm.create_schedule(outputs.op)

    schedule_yolo_conv_x86(s, outputs, inputs, weight, bias)

    arg_bufs = [inputs, weight, bias, outputs]
    stmt = tvm.lower(s, arg_bufs, simple_mode=True)
    print(stmt)
    dev_id = 0
    time_cost = _evaluate(s, arg_bufs, "llvm", dev_id, number=number)
    print("Yolo conv24 use", time_cost, "ms")

    """ For pytorch """
    out_channel, in_channel, kernel_height, kernel_width = yolo_conv.weight_shape
    padding, stride, dilation, groups = (yolo_conv.padding, yolo_conv.stride, yolo_conv.dilation, yolo_conv.groups)
    conv2d_torch = torch.nn.Conv2d(in_channel, out_channel, (kernel_height, kernel_width), padding=padding, stride=stride, dilation=dilation, groups=groups)
    
    # warm up
    inputs = torch.rand(batch_size, *input_shape)
    res = conv2d_torch(inputs)

    times = time.time()
    for _ in range(number):
        res = conv2d_torch(inputs)
    times = time.time() - times
    print("Pytorch on cpu use: {}ms".format(times / number * 1e3))

    # to test the correctness, currently the result is wrong becasue of the schedule
    # if you change line 148 to 'outer = s[write_cache].fuse(gemm_g, gemm_go)'
    # the result is correct
    ctx = tvm.context("llvm", 0)
    inputs_np = np.random.random(inputs.shape).astype("float32") * 100
    weight_np = np.random.random(to_tuple(weight.shape)).astype(weight.dtype) * 100
    outputs_np = np.zeros(shape=to_tuple(outputs.shape), dtype=np.float32) 
    bias_np = np.random.random(size=to_tuple(bias.shape)).astype(bias.dtype) * 100

    inputs_tvm = tvm.nd.array(inputs_np, ctx)
    weight_tvm = tvm.nd.array(weight_np, ctx)
    outputs_tvm = tvm.nd.array(outputs_np, ctx)
    bias_tvm = tvm.nd.array(bias_np, ctx)

    inputs_torch = torch.tensor(inputs_np)
    weight_torch = torch.tensor(weight_np)
    bias_torch = torch.tensor(bias_np)

    func_tvm = tvm.build(s, arg_bufs, "llvm")
    func_tvm(inputs_tvm, weight_tvm, bias_tvm, outputs_tvm)
    outputs_torch = torch.nn.functional.conv2d(inputs_torch, weight_torch, bias=bias_torch, padding=padding, stride=stride, dilation=dilation, groups=groups)
    the_same = test_allclose(outputs_tvm.asnumpy(), outputs_torch.numpy(), rtol=1e-5, print_diff=True)
    if the_same:
        print("The same!")
    else:
        print("Not the same!")

"""
Result:
    batch: 1
        Yolo conv24 use 0.5344364 ms
        Pytorch on cpu use: 1.259779930114746ms
    batch: 2
        Yolo conv24 use 1.9315735 ms
        Pytorch on cpu use: 1.2192821502685547ms
    batch: 8
        Yolo conv24 use 7.96527069 ms
        Pytorch on cpu use: 2.0935559272766113ms
    
    batch: 16
        Yolo conv24 use 15.24043944 ms
        Pytorch on cpu use: 2.525360584259033ms
    batch: 32
        Yolo conv24 use 38.23161417000001 ms
        Pytorch on cpu use: 8.913915157318115ms
    batch: 64
        Yolo conv24 use 101.92375082999999 ms
        Pytorch on cpu use: 15.264289379119873ms
"""

if __name__ == "__main__":
    try_yolo_conv(batch_size=64)