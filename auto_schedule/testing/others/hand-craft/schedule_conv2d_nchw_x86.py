"""
High performance schedule for conv2d_nchw
Target x86 CPU

====================================
**Author**: `Size Zheng`
"""
import tvm 
from auto_schedule.measure import _evaluate
from auto_schedule.nn import SqueezeNetFire8


def schedule_yolo_conv_x86(s, outputs, inputs, weight):
    return 
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
    rc, ry, rx = s[write_cache].op.reduce_axis
    rco, rci = s[write_cache].split(rc, nparts=rc_factors[0])
    rcm, rci = s[write_cache].split(rci, nparts=rc_factors[1])
    rxo, rxi = s[write_cache].split(rx, nparts=rx_factors[0])
    rxm, rxi = s[write_cache].split(rxi, nparts=rx_factors[1])
    ryo, ryi = s[write_cache].split(ry, nparts=ry_factors[0])
    rym, ryi = s[write_cache].split(ryi, nparts=ry_factors[1])

    # reorder
    s[write_cache].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, wb, wk, wp, wq)

    
    s[outputs].pragma(outer, 'auto_unroll_max_step', 1500)
    # s[write_cache].vectorize(s[write_cache].op.axis[-1])

    s[padded].compute_inline()


def try_yolo_conv(batch_size=1):
    # get the compute
    yolo_conv = SqueezeNetFire8()
    input_shape = yolo_conv.get_intput_shape()
    inputs = tvm.placeholder((batch_size, *input_shape), dtype="float32")
    weight = yolo_conv.get_weight()
    outputs = yolo_conv(inputs)
    bias = yolo_conv.get_bias()
    
    s = tvm.create_schedule(outputs.op)
    # schedule_yolo_conv_x86(s, outputs, inputs, weight)

    arg_bufs = [inputs, weight, bias, outputs]
    stmt = tvm.lower(s, arg_bufs, simple_mode=True)
    print(stmt)
    dev_id = 1
    time_cost = _evaluate(s, arg_bufs, "llvm", dev_id, 10)
    print("Yolo conv24 use", time_cost, "ms")


if __name__ == "__main__":
    try_yolo_conv(batch_size=1)