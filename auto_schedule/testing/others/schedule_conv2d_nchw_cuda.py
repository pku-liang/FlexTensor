"""
High performance schedule for conv2d_nchw
Target NVIDIA GPU P100

====================================
**Author**: `Size Zheng`
"""
import tvm 
from auto_schedule.measure import _evaluate
from auto_schedule.testing.layers import YoloConvLayer17


def schedule_yolo_conv_cuda(s, outputs, inputs, weight):
    # inline the padding operation
    padded = outputs.op.input_tensors[0]
    

    # create cache
    write_cache = s.cache_write(outputs, "local")
    read_share_weight = s.cache_read(weight, "shared", [write_cache])
    # read_local_weight = s.cache_read(read_share_weight, "local", [write_cache])
    read_share_inputs = s.cache_read(padded, "shared", [write_cache])
    # read_local_inputs = s.cache_read(read_share_inputs, "local", [write_cache])

    # tunable parameters
    b_factors = [2, 4, 1, 1]
    k_factors = [8, 4, 8, 2]
    p_factors = [7, 1, 2, 1]
    q_factors = [1, 1, 14, 1]
    rc_factors = [1, 32, 32]         # outer-->inner
    ry_factors = [1, 1, 1]
    rx_factors = [1, 1, 1]

    # prepare thread_axis
    bx = tvm.thread_axis("blockIdx.x")
    by = tvm.thread_axis("blockIdx.y")
    bz = tvm.thread_axis("blockIdx.z")
    vx = tvm.thread_axis("vthread")
    vy = tvm.thread_axis("vthread")
    vz = tvm.thread_axis("vthread")
    tx = tvm.thread_axis("threadIdx.x")
    ty = tvm.thread_axis("threadIdx.y")
    tz = tvm.thread_axis("threadIdx.z")

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

    tbo, bi = s[outputs].split(bi, nparts=b_factors[2])
    tko, ki = s[outputs].split(ki, nparts=k_factors[2])
    tpo, pi = s[outputs].split(pi, nparts=p_factors[2])
    tqo, qi = s[outputs].split(qi, nparts=q_factors[2])

    # reorder
    s[outputs].reorder(bo, ko, po, qo, vbo, vko, vpo, vqo, tbo, tko, tpo, tqo, bi, ki, pi, qi)

    kernel_scope, bo = s[outputs].split(bo, nparts=1)

    # fuse
    bko = s[outputs].fuse(bo, ko)
    vbko = s[outputs].fuse(vbo, vko)
    tbko = s[outputs].fuse(tbo, tko)
    bki = s[outputs].fuse(bi, ki)

    # bind
    s[outputs].bind(bko, bz)
    s[outputs].bind(po, by)
    s[outputs].bind(qo, bx)
    s[outputs].bind(vbko, vz)
    s[outputs].bind(vpo, vy)
    s[outputs].bind(vqo, vx)
    s[outputs].bind(tbko, tz)
    s[outputs].bind(tpo, ty)
    s[outputs].bind(tqo, tx)

    # compute at write cache
    s[write_cache].compute_at(s[outputs], tqo)

    # rfactor
    rc, ry, rx = s[write_cache].op.reduce_axis
    rco, rci = s[write_cache].split(rc, nparts=rc_factors[0])
    rcm, rci = s[write_cache].split(rci, nparts=rc_factors[1])
    ryo, ryi = s[write_cache].split(ry, nparts=ry_factors[0])
    rym, ryi = s[write_cache].split(ryi, nparts=ry_factors[1])
    rxo, rxi = s[write_cache].split(rx, nparts=rx_factors[0])
    rxm, rxi = s[write_cache].split(rxi, nparts=rx_factors[1])
    s[write_cache].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, *s[write_cache].op.axis)

    # compute at read cache
    s[read_share_weight].compute_at(s[write_cache], rxm)
    # s[read_local_weight].compute_at(s[write_cache], rxi)
    s[read_share_inputs].compute_at(s[write_cache], rxm)
    # s[read_local_inputs].compute_at(s[write_cache], rxi)

    # cooperative fetching
    for cache in [read_share_inputs, read_share_weight]:
        cb, ck, ch, cw = s[cache].op.axis
        fused = s[cache].fuse(cb, ck, ch, cw)
        bindz, fused = s[cache].split(fused, nparts=k_factors[2])
        bindy, fused = s[cache].split(fused, nparts=p_factors[2])
        bindx, fused = s[cache].split(fused, nparts=q_factors[2])
        s[cache].bind(bindx, tx)
        s[cache].bind(bindy, ty)
        s[cache].bind(bindz, tz)
    
    s[outputs].pragma(kernel_scope, 'auto_unroll_max_step', 1500)
    s[outputs].pragma(kernel_scope, 'unroll_explicit', 1)

    s[padded].compute_inline()


def try_yolo_conv(batch_size=1):
    # get the compute
    yolo_conv = YoloConvLayer17()
    input_shape = yolo_conv.get_intput_shape()
    inputs = tvm.placeholder((batch_size, *input_shape), dtype="float32")
    weight = yolo_conv.get_weight()
    outputs = yolo_conv(inputs)
    
    s = tvm.create_schedule(outputs.op)
    schedule_yolo_conv_cuda(s, outputs, inputs, weight)

    arg_bufs = [inputs, weight, outputs]
    stmt = tvm.lower(s, arg_bufs, simple_mode=True)
    print(stmt)
    dev_id = 0
    ctx = tvm.nd.context("cuda", dev_id)
    max_dims = ctx.max_thread_dimensions
    kwargs = {
        "max_shared_memory_per_block": ctx.max_shared_memory_per_block,
        "max_threads_per_block": ctx.max_threads_per_block,
        "max_thread_x": max_dims[0],
        "max_thread_y": max_dims[1],
        "max_thread_z": max_dims[2]
    }
    verify = tvm.ir_pass.VerifyGPUCode(stmt, kwargs)
    print(verify)
    func = tvm.build(s, arg_bufs, "cuda")
    print(func.import_modules)
    time_cost = _evaluate(s, arg_bufs, "cuda", dev_id, 100)
    print("Yolo conv24 use", time_cost, "ms")


if __name__ == "__main__":
    try_yolo_conv(batch_size=8)