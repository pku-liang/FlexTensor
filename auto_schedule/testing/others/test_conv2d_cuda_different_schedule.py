"""
Test different schedule on conv2d_nchw
Target NVIDIA GPU

====================================
**Author**: `Size Zheng`
"""
import tvm 
from auto_schedule.measure import _evaluate
from auto_schedule.nn import conv2d_nchw
from auto_schedule.configs.conv2d_config import yolo_shapes_b8
from auto_schedule.utils import any_factor_split


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
    # b_factors = [2, 4, 1, 1]
    # k_factors = [8, 4, 8, 2]
    # p_factors = [7, 1, 2, 1]
    # q_factors = [1, 1, 14, 1]
    # rc_factors = [1, 32, 32]         
    # ry_factors = [1, 1, 1]
    # rx_factors = [1, 1, 1]
    b_factors = [2, 2, 2, 1]
    k_factors = [8, 4, 8, 2]
    p_factors = [7, 1, 2, 1]
    q_factors = [1, 1, 14, 1]
    rc_factors = [1, 32, 32]         
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

    kernel_scope, b = s[outputs].split(b, nparts=1)

    # bk = s[outputs].fuse(b, k)
    # bko, bki = s[outputs].split(bk, nparts=b_factors[0] * k_factors[0])
    # po, pi = s[outputs].split(p, nparts=p_factors[0])
    # qo, qi = s[outputs].split(q, nparts=q_factors[0])

    # vbko, bki = s[outputs].split(bki, nparts=b_factors[1] * k_factors[1])
    # vpo, pi = s[outputs].split(pi, nparts=p_factors[1])
    # vqo, qi = s[outputs].split(qi, nparts=q_factors[1])

    # tbko, bki = s[outputs].split(bki, nparts=b_factors[2] * k_factors[2])
    # tpo, pi = s[outputs].split(pi, nparts=p_factors[2])
    # tqo, qi = s[outputs].split(qi, nparts=q_factors[2])

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
    s[outputs].reorder(po, bo, ko, qo, vqo, vbo, vko, vpo, tbo, tko, tpo, tqo, bi, ki, pi, qi)
    # s[outputs].reorder(po, bko, qo, vqo, vbko, vpo, tbko, tpo, tqo, bki, pi, qi)

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

    rc, ry, rx = s[write_cache].op.reduce_axis
    rco, rci = s[write_cache].split(rc, nparts=rc_factors[0])
    rcm, rci = s[write_cache].split(rci, nparts=rc_factors[1])
    ryo, ryi = s[write_cache].split(ry, nparts=ry_factors[0])
    rym, ryi = s[write_cache].split(ryi, nparts=ry_factors[1])
    rxo, rxi = s[write_cache].split(rx, nparts=rx_factors[0])
    rxm, rxi = s[write_cache].split(rxi, nparts=rx_factors[1])
    a, b, c, d = s[write_cache].op.axis
    s[write_cache].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, a, b, c, d)

    # compute at read cache
    s[read_share_weight].compute_at(s[write_cache], rxm)
    # s[read_local_weight].compute_at(s[write_cache], rxi)
    s[read_share_inputs].compute_at(s[write_cache], rxm)
    # s[read_local_inputs].compute_at(s[write_cache], rxi)

    # cooperative fetching
    for cache in [read_share_inputs, read_share_weight]:
        cb, ck, ch, cw = s[cache].op.axis
        fused = s[cache].fuse(cb, ck, ch, cw)
        fused, bindx = s[cache].split(fused, factor=q_factors[2])
        fused, bindy = s[cache].split(fused, factor=p_factors[2])
        fused, bindz = s[cache].split(fused, factor=b_factors[2] * k_factors[2])       
        
        s[cache].bind(bindx, tx)
        s[cache].bind(bindy, ty)
        s[cache].bind(bindz, tz)
    
    s[outputs].pragma(kernel_scope, 'auto_unroll_max_step', 1500)
    s[outputs].pragma(kernel_scope, 'unroll_explicit', 1)

    s[padded].compute_inline()



def try_yolo_conv(config):
    # get the compute
    # (1, 3, 448, 448, 64, 3, 7, 7, 1, 2, 3, 1, 1)
    batch, CI, H, W, CO, _, kh, kw, _, st, pad, dilation, group = config
    inputs = tvm.placeholder((batch, CI, H, W), dtype="float32")
    weight = tvm.placeholder((CO, CI, kh, kw), dtype="float32")
    outputs = conv2d_nchw(inputs, weight, stride=st, padding=pad, dilation=dilation, groups=group)
    
    s = tvm.create_schedule(outputs.op)
    schedule_yolo_conv_cuda(s, outputs, inputs, weight)

    arg_bufs = [inputs, weight, outputs]
    stmt = tvm.lower(s, arg_bufs, simple_mode=True)
    # print(stmt)
    dev_id = 3
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
    print("config is:\n %s" % (str(config)))
    if verify:
        print("Valid kernel")
        time_cost = _evaluate(s, arg_bufs, "cuda", dev_id, 10)
        print("Yolo conv use", time_cost, "ms\n")
    else:
        print("Invalid kernel")
        time_cost = float("inf")
    return time_cost


if __name__ == "__main__":
    res = []
    for config in yolo_shapes_b8[10:11]:
        cost = try_yolo_conv(config)
        res.append(cost)
    for ele in res:
        print(ele)
