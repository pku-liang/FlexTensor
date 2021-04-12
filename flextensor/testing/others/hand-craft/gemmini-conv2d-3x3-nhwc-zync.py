import tvm
import numpy as np
import os
import sys
import logging

from tvm import micro, rpc
from tvm.contrib import util
from tvm.micro.base import compile_micro_mod
from topi.util import get_const_tuple
from types import SimpleNamespace

dim = 128

if len(sys.argv) == 2:
    if sys.argv[1] == "host":  # run on x86
        cpu = True
        micro_device_config = micro.device.host.default_config()
    elif sys.argv[1] == 'cpu':  # run on RISC-V without Gemmini
        cpu = True
    else:
        dim = int(sys.argv[1])

N = 1
RC = 128
K = 128
P = 28
Q = 28
R = 3
S = 3
H = (P+R//2*2)
W = (Q+S//2*2)

zynq_host = '127.0.0.1'
micro_device_config = micro.device.riscv.rocketchip_zynq.default_config(0x800000000, zynq_host, 6666)
rpc_server = '127.0.0.1'
rpc_port = 9190
device_key = 'gemmini'

# Working code generation for Gemmini
# Matrix multiplication without padding / bias
# Overflow support seems weird, use small numbers

target = 'c -device=micro_dev'
tune = False
cpu = False
verify = True

DIM = 16
dtype = "int8"
ti = np.iinfo(dtype)
min_input = -2
max_input = 2


def conv2d_rounding(A, B):
    """
    chunked conv2d which converts datatypes and filters values,
    converting > 127 to 127 and < -128 to -128
    """
    # C = np.zeros((N, P, Q, K)).astype("int32")  # output
    # for b in range(N):
    #     for p in range(P):
    #         for q in range(Q):
    #             for k in range(K):
    #                 for rc in range(RC):
    #                     for rr in range(R):
    #                         for rs in range(S):
    #                             C[b, p, q, k] += A[b, p+rr, q+rs, rc] * B[rr, rs, rc, k]
    import torch
    A = torch.tensor(A, dtype=torch.int32).permute(0, 3, 1, 2)
    B = torch.tensor(B, dtype=torch.int32).permute(3, 2, 0, 1)
    C = torch.nn.functional.conv2d(A, B, bias=None, stride=1, padding=0, dilation=1, groups=1)
    C = C.permute(0, 2, 3, 1).numpy()
    C[C > 127] = 127
    C[C < -128] = -128
    return C.astype(np.int8)


def print_nested(l, f):
    with open(f, 'w') as ff:
        for ll in l:
            print(' '.join(map(lambda i: '%2d' % i, ll)), file=ff)


def conv2d(N, RC, H, W, K, P, Q, R, S, dtype):
    A = tvm.placeholder((N, H, W, RC), name='A', dtype=dtype)
    B = tvm.placeholder((R, S, RC, K), name='B', dtype=dtype)
    rc = tvm.reduce_axis((0, RC), name='rc')
    rr = tvm.reduce_axis((0, R), name="rr")
    rs = tvm.reduce_axis((0, S), name="rs")
    C = tvm.compute((N, P, Q, K), lambda b, p, q, k:
                    tvm.sum(A[b, p+rr, q+rs, rc] * B[rr, rs, rc, k], axis=[rr, rs, rc]), name='C')
    s = tvm.create_schedule(C.op)
    b, p, q, k = C.op.axis
    rr, rs, rc = C.op.reduce_axis

    b_factors = [1, 1]  # 1
    k_factors = [4, 32]  # 128
    p_factors = [1, 28]  # 28
    q_factors = [4, 32]  # 128
    rc_factors = [1, 28]  # 28
    rr_factors = [3, 1]  # 1
    rs_factors = [3, 1]  # 1

    bo, bi = s[C].split(b, factor=b_factors[1])
    ko, ki = s[C].split(k, factor=k_factors[1])
    po, pi = s[C].split(p, factor=p_factors[1])
    qo, qi = s[C].split(q, factor=q_factors[1])
    rco, rci = s[C].split(rc, factor=rc_factors[1])
    rro, rri = s[C].split(rr, factor=rr_factors[1])
    rso, rsi = s[C].split(rs, factor=rs_factors[1])

    s[C].reorder(bo, ko, po, qo, bi, pi, rro, rso, rco, rri, rsi, qi, ki, rci)

    # use a different kernel size for the last iteration if
    # the input size is not divisible by the tiling factor
    last_n = Q % q_factors[1]
    nc = tvm.expr.EQ(qo, Q // q_factors[1]) if last_n != 0 else False
    last_n = last_n if last_n != 0 else q_factors[1]

    last_l = RC % rc_factors[1]
    lc = tvm.expr.EQ(rco, RC // rc_factors[1]) if last_l != 0 else False
    last_l = last_l if last_l != 0 else rc_factors[1]

    last_m = K % k_factors[1]
    mc = tvm.expr.EQ(ko, K // k_factors[1]) if last_m != 0 else False
    last_m = last_m if last_m != 0 else k_factors[1]

    print(q_factors[1], k_factors[1], rc_factors[1])
    gemm = intrinsic_gemm(q_factors[1], k_factors[1], rc_factors[1], last_n,
                          last_m, last_l, nc, mc, lc)
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    s[C].tensorize(qi, gemm)
    #s[C].pragma(xo, "prologue", "do_flush")
    s[C].pragma(bo, "epilogue", "do_fence")
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    return s, [A, B, C]


def intrinsic_gemm(i, j, k, il, jl, kl, ic, jc, kc):
    """
    (i, k) * (k, j)
    i, j, k: normal iteration size
    il, jl, kl: last iteration size
    ic, jc, kc: last iteration condition
    """
    assert i * k + k * j <= 256 * 1024, 'input too large for scratchpad'
    assert 4 * (i * j) <= 64 * 1024, 'input too large for accumulator'

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


def main():
    intrinsic_filename = 'kernel.c'
    aux_sources = [f"{os.path.realpath(intrinsic_filename)}"]
    aux_options = [f"-I{os.path.dirname(os.path.realpath(intrinsic_filename))}"]
    s, bufs = conv2d(N, RC, H, W, K, P, Q, R, S, dtype)
    print("Build function...")
    func = tvm.build(s, bufs, target=target)

    print(func.get_source())

    tmp_dir = util.tempdir()
    mod_path = tmp_dir.relpath('micro.obj')
    compile_micro_mod(mod_path,
                    func, micro_device_config,
                    aux_sources=aux_sources,
                    aux_options=aux_options)

    tracker = rpc.connect_tracker(rpc_server, rpc_port)
    remote = tracker.request(device_key, priority=1,
                             session_timeout=10000)
    
    print('Uploading', mod_path)
    remote.upload(mod_path)
    func = remote.load_module(os.path.split(mod_path)[1])
    ctx = remote.device(target, 0)

    time_f = func.time_evaluator(func.entry_name, ctx)

    print('Creating buffers on device')
    A_np = np.random.uniform(min_input, max_input, [N, H, W, RC]).astype(dtype)
    B_np = np.random.uniform(min_input, max_input, [R, S, RC, K]).astype(dtype)
    # print("A input")
    # print(A_np)
    # print("B input")
    # print(B_np)

    a = tvm.nd.array(A_np, ctx, copy=verify)
    b = tvm.nd.array(B_np, ctx, copy=verify)
    c = tvm.nd.empty([N, P, Q, K], dtype, ctx)

    print('Running function on device')
    costs = time_f(a, b, c).results
    print('Best result cost:', costs)

    if verify:
        print('Checking correctness...')
        golden = conv2d_rounding(A_np, B_np)
        tvm.testing.assert_allclose(c.asnumpy(), golden)

        print('Micro correctness check passed.')
    else:
        print('Micro correctness check skipped.')


if __name__ == "__main__":
    main()