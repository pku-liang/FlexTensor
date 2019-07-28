"""
Test different schedule on conv2d_nchw
Target CPU

====================================
**Author**: `Size Zheng`
"""
import tvm 
from auto_schedule.measure import _evaluate
from auto_schedule.nn import YoloConvLayer17
from auto_schedule.utils import any_factor_split



class Config(object):
    def __init__(self):
        self.b_factors = [1, 1, 1, 1]
        self.k_factors = [8, 4, 8, 2]
        self.p_factors = [7, 1, 2, 1]
        self.q_factors = [1, 1, 14, 1]
        self.rc_factors = [1, 32, 32]         
        self.ry_factors = [1, 1, 1]
        self.rx_factors = [1, 1, 1]

    def __str__(self):
        ret = []
        ret.append("b_factors:" + str(self.b_factors) + "\n")
        ret.append("k_factors:" + str(self.k_factors) + "\n")
        ret.append("p_factors:" + str(self.p_factors) + "\n")
        ret.append("q_factors:" + str(self.q_factors) + "\n")
        ret.append("rc_factors:" + str(self.rc_factors) + "\n")
        ret.append("ry_factors:" + str(self.ry_factors) + "\n")
        ret.append("rx_factors:" + str(self.rx_factors) + "\n")
        return "".join(ret)


def schedule_yolo_conv_llvm(s, outputs, inputs, weight, config):
    # inline the padding operation
    padded = outputs.op.input_tensors[0]
    

    # create cache
    write_cache = s.cache_write(outputs, "local")
    read_share_weight = s.cache_read(weight, "shared", [write_cache])
    # read_local_weight = s.cache_read(read_share_weight, "local", [write_cache])
    read_share_inputs = s.cache_read(padded, "shared", [write_cache])
    # read_local_inputs = s.cache_read(read_share_inputs, "local", [write_cache])

    # tunable parameters
    b_factors = config.b_factors
    k_factors = config.k_factors
    p_factors = config.p_factors
    q_factors = config.q_factors
    rc_factors = config.rc_factors     
    ry_factors = config.ry_factors
    rx_factors = config.rx_factors

    # split the spatial axes
    b, k, p, q = s[outputs].op.axis

    kernel_scope, b = s[outputs].split(b, nparts=1)

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

    # fuse
    bko = s[outputs].fuse(bo, ko)
    vbko = s[outputs].fuse(vbo, vko)
    tbko = s[outputs].fuse(tbo, tko)
    bki = s[outputs].fuse(bi, ki)

    outer = s[outputs].fuse(bko, po, qo)
    s[outputs].parallel(outer)

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
    
    s[outputs].pragma(kernel_scope, 'auto_unroll_max_step', 1500)
    s[outputs].pragma(kernel_scope, 'unroll_explicit', 1)

    s[padded].compute_inline()


__COUNTER__ = 0


def try_yolo_conv(batch_size, config):
    global __COUNTER__
    __COUNTER__ += 1
    # get the compute
    yolo_conv = YoloConvLayer17()
    input_shape = yolo_conv.get_intput_shape()
    inputs = tvm.placeholder((batch_size, *input_shape), dtype="float32")
    weight = yolo_conv.get_weight()
    outputs = yolo_conv(inputs)
    
    s = tvm.create_schedule(outputs.op)
    schedule_yolo_conv_llvm(s, outputs, inputs, weight, config)

    arg_bufs = [inputs, weight, outputs]
    stmt = tvm.lower(s, arg_bufs, simple_mode=True)
    # print(stmt)
    dev_id = 0
    time_cost = _evaluate(s, arg_bufs, "llvm", dev_id, 10)
    print("Yolo conv17 use", time_cost, "ms\n")
    return time_cost


if __name__ == "__main__":
    import random
    config = Config()
    # k_split_lst = list(filter(lambda x: x[1] == 4 and x[3] == 2, any_factor_split(512, 4)))
    # print(k_split_lst)
    # print(len(k_split_lst))
    rc_split_lst = list(filter(lambda x: x[1] == 1, any_factor_split(1024, 3)))
    flop = 14 * 14 * 512 * (1024 + 1023)
    record = []
    for ele in rc_split_lst:
        # config.k_factors = ele
        config.rc_factors = ele
        time_cost = try_yolo_conv(1, config)
        record.append((ele, flop / (time_cost / 1e3) / 1e9))
    for ele in record:
        print(ele[0][2])
    for ele in record:
        print(ele[1])
