from __future__ import absolute_import

import tvm
from flextensor.task import register
from flextensor.measure import _evaluate
from flextensor.nn import conv2d_nchwc
from flextensor.configs.conv2d_config import yolo_shapes_b1


def conv2d_nchwc_compute_avx2(N, C, H, W, K, k=3, use_bias=False, st=1, pad=0, dilation=1, group=1):
    vlen = 8    # AVX2 provides 256 bit operations
    inputs = tvm.placeholder([N, C // vlen // group, H, W, vlen], dtype="float32")
    weight = tvm.placeholder([K // vlen, C // vlen // group, k, k, vlen, vlen], dtype="float32")
    if use_bias:
        bias = tvm.placeholder([K // vlen, vlen], dtype="float32")
    else:
        bias = None 
    output = conv2d_nchwc(inputs, weight, bias, stride=st, padding=pad, dilation=dilation, groups=group)
    return output 


h_factors = [56, 1]
w_factors = [28, 2]
ic_factors = [2, 16]


def conv2d_nchwc_schedule_avx2_yolo_conv6(output):
    s = tvm.create_schedule(output.op)
    pad = s[output].op.input_tensors[0]
    s[pad].compute_inline()

    cache_write = s.cache_write(output, "global")

    # tunable parameters
    fuse_level = 2
    
    unroll_depth = 2500

    # split spatial axes
    n, ko, h, w, ki = s[output].op.axis
    ho, hi = s[output].split(h, nparts=h_factors[0])
    wo, wi = s[output].split(w, nparts=w_factors[0])
    
    # reorder
    s[output].reorder(n, ko, ho, hi, wo, wi, ki)

    # parallel
    candidate = [ko, ho, hi, wo]
    to_fuse = [candidate[i] for i in range(min(fuse_level, len(candidate)))]
    fused = s[output].fuse(*to_fuse)
    s[output].parallel(fused)
    fuse_to_end = fuse_level >= len(candidate)

    # vectorize
    s[output].vectorize(ki)

    # unroll
    # s[output].pragma(fused, 'auto_unroll_max_step', unroll_depth)

    # compute at
    if fuse_to_end:
        s[cache_write].compute_at(s[output], fused)
    else:
        s[cache_write].compute_at(s[output], wo)

    # split spatial axes
    n, ko, h, w, ki = s[cache_write].op.axis
    wo, wi = s[cache_write].split(w, nparts=w_factors[0])

    # reduce axes 
    rc, rp, rq = s[cache_write].op.reduce_axis
    rco, rci = s[cache_write].split(rc, nparts=ic_factors[0])

    # reorder 
    s[cache_write].reorder(ko, h, wo, rco, rp, rci, rq, wi, ki)

    # unroll
    s[cache_write].unroll(rq)
    s[cache_write].unroll(wi)

    # fuse
    s[cache_write].fuse(ko, h)

    # vectorize
    s[cache_write].vectorize(ki)

    return s


if __name__ == "__main__":
    N, C, H, W, K, _, k, _, _, st, pad, dilation, group = yolo_shapes_b1[5]

    use_bias = False
    vlen = 8
    target = "llvm"
    dev_id = 0
    args = (N, C, H, W, K, k, use_bias, st, pad, dilation, group)
    register(conv2d_nchwc_compute_avx2, "conv2d_nchwc", "yolo_conv6", args, target, dev_id=dev_id)

    for i in [1, 2, 4, 8, 16, 32]:
        ic_factors[0] = i
        ic_factors[1] = 256 // i
        # get compute
        inputs = tvm.placeholder([N, C // vlen // group, H, W, vlen], dtype="float32")
        weight = tvm.placeholder([K // vlen, C // vlen // group, k, k, vlen, vlen], dtype="float32")
        if use_bias:
            bias = tvm.placeholder([K // vlen, vlen], dtype="float32")
        else:
            bias = None 
        output = conv2d_nchwc(inputs, weight, bias, stride=st, padding=pad, dilation=dilation, groups=group)
        s = conv2d_nchwc_schedule_avx2_yolo_conv6(output)
        time_cost = _evaluate(s, [inputs, weight, output], target, dev_id, 10)
        print("Run time: %f ms, throughput: %f GFLOPS" % (time_cost, N * C * H * W * K * k * k / st / st / group / 1e6 / time_cost))
