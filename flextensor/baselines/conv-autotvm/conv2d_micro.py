import tvm
import numpy as np
import os
import sys
import logging

from tvm import micro, autotvm
from tvm.contrib import util
from tvm.micro.base import compile_micro_mod
from topi.util import get_const_tuple
from types import SimpleNamespace
import math
import json
from itertools import permutations
from flextensor.model_measure import ModelBuilder, ModelRunner
from flextensor.utils import get_iter_info
from flextensor.ppa_model import measure_latency, set_config_file
from tvm.autotvm.measure.measure import MeasureResult, MeasureErrorNo

shape = (1, 64, 56, 56, 64, 3, 3, 1, 1, 1)
# shape = (1, 64, 56, 56, 256, 1, 1, 1, 0, 1)  # res2a_branch1


zynq_host = '127.0.0.1'
micro_device_config = micro.device.riscv.rocketchip_zynq.default_config(
    0x800000000, zynq_host, 6666)
rpc_server = '127.0.0.1'
rpc_port = 9190
device_key = 'gemmini'
# log_filename = f'conv2d-model-{"x".join(map(str, shape))}.log'
log_filename = "conv2d.log"

# Working code generation for Gemmini
# Matrix multiplication without padding / bias
# Overflow support seems weird, use small numbers

target = 'c -device=micro_dev'
tune = True
cpu = False
verify = True

# alternative config
# DIM = 8
# sp_kb = 128
# acc_kb = 32

# default config
DIM = 16
sp_kb = 256
acc_kb = 64


dtype = "int8"
ti = np.iinfo(dtype)
# min_input = ti.min
# max_input = ti.max
min_input = -8
max_input = 8


def print_nested(l, f):
    with open(f, 'w') as ff:
        for ll in l:
            print(' '.join(map(lambda i: '%2d' % i, ll)), file=ff)


def gen_pow2_factors(dim):
    low, high = int(math.log2(DIM)), math.ceil(math.log2(dim)) + 1
    return list(map(lambda i: 2 ** i, range(low, high)))


def gen_factors(x):
    return [i for i in range(1, x + 1) if x % i == 0]


ORDERS = [
    ("n", "rh", "rw", "h", "wo", "rco", "ko"),
    ("n", "h", "wo", "ko", "rh", "rw", "rco"),
    ("n", "rh", "rw", "rco", "h", "wo", "ko"),
    ("n", "h", "rh", "rw", "wo", "rco", "ko"),
    ("n", "h", "rh", "wo", "rw", "ko", "rco"),
    ("n", "ko", "rco", "wo", "rh", "rw", "h"),
    ("n", "rco", "rw", "rh", "ko", "wo", "h"),
    ("n", "ko", "wo", "h", "rco", "rw", "rh"),
    ("n", "ko", "rco", "wo", "rw", "rh", "h"),
    ("n", "rco", "ko", "rw", "wo", "rh", "h"),
]


@autotvm.template
def conv2d(N, C, H, W, K, Hk, Wk, S, P, D, dtype='int8'):
    H += 2 * P
    W += 2 * P
    Ho = (H - (Hk - 1) * D - 1) // S + 1
    Wo = (W - (Wk - 1) * D - 1) // S + 1
    I = tvm.placeholder((N, H, W, C), name="I", dtype=dtype)
    F = tvm.placeholder((Hk, Wk, C, K), name="F", dtype=dtype)
    rh = tvm.reduce_axis((0, Hk), name="rh")
    rw = tvm.reduce_axis((0, Wk), name="rw")
    rc = tvm.reduce_axis((0, C), name="rc")
    O = tvm.compute(
        (N, Ho, Wo, K),
        lambda n, h, w, k: tvm.sum(
            I[n, h + rh, w + rw, rc] * F[rh, rw, rc, k],
            axis=[rh, rw, rc]
        ), name="O")

    s = tvm.create_schedule(O.op)
    cfg = autotvm.get_config()

    cfg.define_knob('w_factor', gen_pow2_factors(Wo))
    cfg.define_knob('c_factor', gen_pow2_factors(C))
    cfg.define_knob('k_factor', gen_pow2_factors(K))
    cfg.define_knob('order_index', list(range(len(ORDERS))))

    w_f = cfg['w_factor'].val
    c_f = cfg['c_factor'].val
    k_f = cfg['k_factor'].val
    ord_idx = cfg['order_index'].val

    n, h, w, k = s[O].op.axis
    rh, rw, rc = s[O].op.reduce_axis

    wo, wi = s[O].split(w, factor=w_f)
    rco, rci = s[O].split(rc, factor=c_f)
    ko, ki = s[O].split(k, factor=k_f)

    STR2AXIS = {
        "n": n, "h": h, "wo": wo, "ko": ko, "rh": rh, "rw": rw, "rco": rco
    }

    s[O].reorder(*[STR2AXIS[s] for s in ORDERS[ord_idx]], wi, ki, rci)

    gemm = generate_intrinsic_gemm_int8_dim16(
        Wo, K, C, w_f, k_f, c_f, wo, ko, rco)

    s[O].tensorize(wi, gemm)
    s[O].pragma(n, "epilogue", "do_fence")

    return s, [I, F, O]


def generate_intrinsic_gemm_int8_dim16(N, M, K, fN, fM, fK, axisN, axisM, axisK):
    last_n = N % fN
    nc = tvm.expr.EQ(axisN, N // fN) if last_n != 0 else False
    last_n = last_n if last_n != 0 else fN

    last_l = K % fK
    lc = tvm.expr.EQ(axisK, K // fK) if last_l != 0 else False
    last_l = last_l if last_l != 0 else fK

    last_m = M % fM
    mc = tvm.expr.EQ(axisM, M // fM) if last_m != 0 else False
    last_m = last_m if last_m != 0 else fM

    gemm = intrinsic_gemm(fN, fM, fK, last_n,
                          last_m, last_l, nc, mc, lc)

    return gemm


def intrinsic_gemm(i, j, k, il, jl, kl, ic, jc, kc):
    """
    (i, k) * (k, j)
    i, j, k: normal iteration size
    il, jl, kl: last iteration size
    ic, jc, kc: last iteration condition
    """
    assert i * k + k * j <= sp_kb * 1024, 'input too large for scratchpad'
    assert 4 * (i * j) <= acc_kb * 1024, 'input too large for accumulator'

    a = tvm.placeholder((i, k), name='a', dtype=dtype)
    b = tvm.placeholder((k, j), name='b', dtype=dtype)
    kk = tvm.reduce_axis((0, k), name='k')
    c = tvm.compute((i, j), lambda ii, jj:
                    tvm.sum(a[ii, kk] * b[kk, jj], axis=kk), name='c')

    strideA = tvm.var("sA")
    Ab = tvm.decl_buffer(a.shape, a.dtype,
                         name="A",
                         offset_factor=1,
                         strides=[strideA, 1])
    strideB = tvm.var("sB")
    Bb = tvm.decl_buffer(b.shape, b.dtype,
                         name="B",
                         offset_factor=1,
                         strides=[strideB, 1])
    strideC = tvm.var("sC")
    Cb = tvm.decl_buffer(c.shape, c.dtype,
                         name="C",
                         offset_factor=1,
                         strides=[strideC, 1])

    II = i // DIM + (0 if i % DIM == 0 else 1)
    JJ = j // DIM + (0 if j % DIM == 0 else 1)
    KK = k // DIM + (0 if k % DIM == 0 else 1)
    pad_I = 0 if i % DIM == 0 else (DIM - i % DIM)
    pad_J = 0 if j % DIM == 0 else (DIM - j % DIM)
    pad_K = 0 if k % DIM == 0 else (DIM - k % DIM)

    IIl = il // DIM + (0 if il % DIM == 0 else 1)
    JJl = jl // DIM + (0 if jl % DIM == 0 else 1)
    KKl = kl // DIM + (0 if kl % DIM == 0 else 1)
    pad_Il = 0 if il % DIM == 0 else (DIM - il % DIM)
    pad_Jl = 0 if jl % DIM == 0 else (DIM - jl % DIM)
    pad_Kl = 0 if kl % DIM == 0 else (DIM - kl % DIM)

    II = tvm.if_then_else(ic, IIl, II)
    JJ = tvm.if_then_else(jc, JJl, JJ)
    KK = tvm.if_then_else(kc, KKl, KK)
    pad_I = tvm.if_then_else(ic, pad_Il, pad_I)
    pad_J = tvm.if_then_else(jc, pad_Jl, pad_J)
    pad_K = tvm.if_then_else(kc, pad_Kl, pad_K)

    # reset-update-finalize
    def intrin_func(ins, outs):
        aa, bb = ins
        cc, = outs

        def _body():
            ib = tvm.ir_builder.create()
            # int32_t matmul_kernel(const elem_t *A, const elem_t *B, const acc_t *D,
            #          elem_t *C, int32_t I, int32_t J, int32_t K, int32_t pad_I,
            #          int32_t pad_J, int32_t pad_K, int32_t A_row_len,
            #          int32_t B_row_len, int32_t D_row_len, int32_t C_row_len,
            #          bool no_bias, bool repeating_bias);
            # D is set to a dummy address 1 to determine whether to overwrite
            # accumulator contents: on the first run, 1 will be retained and
            # overwrite the value in the accumulator; on subsequent runs D will be
            # replaced by NULL and C will accumulate on top of the accumulator's contents
            # This is controlled via bit 1 << (ADDR_LEN - 2) - see kernel source
            ib.emit(tvm.call_extern("int32", "matmul_kernel",
                                    aa.access_ptr("r"),
                                    bb.access_ptr("r"),
                                    1,
                                    cc.access_ptr("rw"),
                                    II, JJ, KK,
                                    pad_I, pad_J, pad_K,
                                    strideA, strideB, 0, strideC,
                                    True, False))
            return ib.get()

        def _reset():
            ib = tvm.ir_builder.create()
            # int32_t matmul_reset(elem_t *C, int32_t I, int32_t J, int32_t pad_I,
            #         int32_t pad_J, int32_t C_row_len);
            ib.emit(tvm.call_extern("int32", "matmul_reset",
                                    cc.access_ptr("w"),
                                    II, JJ,
                                    pad_I, pad_J,
                                    strideC))
            return ib.get()

        def _finalize():
            ib = tvm.ir_builder.create()
            # Move out C from accumulator
            # int32_t matmul_finalize(elem_t *C, int32_t I, int32_t J, int32_t pad_I,
            #         int32_t pad_J, int32_t C_row_len);
            ib.emit(tvm.call_extern("int32", "matmul_finalize",
                                    cc.access_ptr("rw"),
                                    II, JJ,
                                    pad_I, pad_J,
                                    strideC))
            return ib.get()
        # standalone (without reduce axis split), reset, update
        return None, _reset(), _body(), _finalize()
    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb}, name="sp_gemm")


def tunning():

    task = autotvm.task.create(conv2d, args=(*shape, dtype), target=target)
    print(task.config_space)

    # define build function
    intrinsic_filename = 'kernel.c'
    aux_sources = [f"{os.path.realpath(intrinsic_filename)}"]
    aux_options = [
        f"-I{os.path.dirname(os.path.realpath(intrinsic_filename))}"]
    if cpu:
        aux_options += ['-DCPU']

    build_func = SimpleNamespace(
        output_format='micro_dev',
        args=[
            micro_device_config,
            aux_sources,
            aux_options,
        ])

    measure_option = autotvm.measure_option(
        builder=ModelBuilder(),
        runner=ModelRunner(),
    )

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    if tune:
        print('Start tuning')
        tuner = autotvm.tuner.GridSearchTuner(task)
        with open(log_filename, "w"):
            pass
        tuner.tune(n_trial=4000,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(log_filename)])

    with autotvm.apply_history_best(log_filename):
        with tvm.target.create(target):

            s, _ = conv2d(*shape, dtype)
            info = get_iter_info(s)
            with open("info.json", "w") as fp:
                json.dump(info, fp)
            cost = measure_latency(get_iter_info(s))

            print('Best result cost:', cost)


def recording(input_file, output_file):
    with open(input_file, mode="r") as log:
        with open(output_file, mode="w") as info:
            for line in log:
                inp, res = autotvm.record.decode(line)
                with autotvm.apply_history_best([(inp, MeasureResult(res.costs, MeasureErrorNo.NO_ERROR, res.all_cost, res.timestamp))]):
                    with tvm.target.create(target):
                        s, _ = conv2d(*shape, dtype)
                        info.write(json.dumps({
                            "info": get_iter_info(s),
                            "cost": np.mean(res.costs) if res.error_no == 0 else None,
                        }) + '\n')


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # set_config_file("micro-2.m")
        tunning()
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "infos.log"
        recording(input_file, output_file)
