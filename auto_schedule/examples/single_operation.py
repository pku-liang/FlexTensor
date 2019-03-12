import tvm
from collections import namedtuple


FUNC_TABLE = {}


def register(args, override=False):
    FuncEntity = namedtuple("FuncEntity", ("func", "args"))

    def do_reg(func):
        if func.__name__ in FUNC_TABLE and not override:
            raise ValueError("Duplicated function name")
        FUNC_TABLE[func.__name__] = FuncEntity(func, args)
    return do_reg


@register(args=(32, 32))
def gemv(N, M, dtype="float32"):
    A = tvm.placeholder((N, M), dtype=dtype, name="A")
    V = tvm.placeholder((M,), dtype=dtype, name="V")
    k = tvm.reduce_axis((0, M), name="k")
    C = tvm.compute((N,), lambda i: tvm.sum(A[i, k] * V[k], axis=k), name="C")
    return C.op, [A, V, C]


@register(args=(32, 32, 32))
def matmul(N, M, L, dtype="float32"):
    A = tvm.placeholder((N, M), dtype=dtype, name="A")
    B = tvm.placeholder((M, L), dtype=dtype, name="B")
    k = tvm.reduce_axis((0, M), name="k")
    C = tvm.compute((N, L), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name="C")

    return C.op, [A, B, C]


@register(args=(25, 10, 100, 10))
def matmul_batch(B, N, M, L, dtype="float32"):
    A = tvm.placeholder((B, N, M), dtype=dtype, name="A")
    mB = tvm.placeholder((M, L), dtype=dtype, name="B")
    k = tvm.reduce_axis((0, M), name="k")
    C = tvm.compute((B, N, L), lambda b, i, j: tvm.sum(A[b, i, k] * mB[k, j], axis=k), name="C")
    return C.op, [A, mB, C]


@register(args=(25, 25, 3, 3, 1, 1))
def conv2d(N, M, K, L, stride=1, padding=0, dtype="float32"):
    A = tvm.placeholder((N, M), dtype=dtype, name="A")
    W = tvm.placeholder((K, L), dtype=dtype, name="W")
    N_out = max(0, (N + padding * 2 - K) // stride) + 1
    M_out = max(0, (M + padding * 2 - L) // stride) + 1
    Apad = tvm.compute((N + 2 * padding, M + 2 * padding),
                       lambda i, j: tvm.if_then_else(tvm.all(i >= padding, j >= padding, i < N + padding, j < M + padding),
                                               A[i - padding, j - padding], 0.0), name="Apad")
    rx, ry = tvm.reduce_axis((0, K), name="rx"), tvm.reduce_axis((0, L), name="ry")
    Output = tvm.compute((N_out, M_out),
                         lambda i, j: tvm.sum(Apad[i * stride + rx, j * stride + ry] * W[rx, ry], axis=[rx, ry]),
                         name="Output")
    return Output.op, [A, W, Output]


@register(args=(16, 32, 32, 5, 5, 1, 2))
def conv2d_batch(B, N, M, K, L, stride=1, padding=0, dtype="float32"):
    A = tvm.placeholder((B, N, M), dtype=dtype, name="A")
    W = tvm.placeholder((K, L), dtype=dtype, name="W")
    N_out = max(0, (N + padding * 2 - K) // stride) + 1
    M_out = max(0, (M + padding * 2 - L) // stride) + 1
    Apad = tvm.compute((B, N + 2 * padding, M + 2 * padding),
                       lambda b, i, j: tvm.if_then_else(tvm.all(i >= padding, j >= padding, i < N + padding, j < M + padding),
                                                  A[b, i - padding, j - padding], 0.0), name="Apad")
    rx, ry = tvm.reduce_axis((0, K), name="rx"), tvm.reduce_axis((0, L), name="ry")
    Output = tvm.compute((B, N_out, M_out),
                         lambda b, i, j: tvm.sum(Apad[b, i * stride + rx, j * stride + ry] * W[rx, ry], axis=[rx, ry]),
                         name="Output")
    return Output.op, [A, W, Output]


@register(args=(32, 32, 3, 5, 5, 1, 2))
def conv2d_channel(N, M, C, K, L, O, stride=1, padding=0, dtype="float32"):
    A = tvm.placeholder((N, M, C), dtype=dtype, name="A")
    W = tvm.placeholder((K, L, C, O), dtype=dtype, name="W")
    N_out = max(0, (N + padding * 2 - K) // stride) + 1
    M_out = max(0, (M + padding * 2 - L) // stride) + 1
    Apad = tvm.compute((N + 2 * padding, M + 2 * padding, C),
                       lambda i, j, k: tvm.if_then_else(tvm.all(i >= padding, j >= padding, i < N + padding, j < M + padding),
                                               A[i - padding, j - padding, k], 0.0), name="Apad")
    rx, ry = tvm.reduce_axis((0, K), name="rx"), tvm.reduce_axis((0, L), name="ry")
    rc = tvm.reduce_axis((0, C), name="rc")
    Output = tvm.compute((N_out, M_out, O),
                         lambda i, j, k: tvm.sum(Apad[i * stride + rx, j * stride + ry, rc] * W[rx, ry, rc, k],
                                                 axis=[rx, ry, rc]),
                         name="Output")
    return Output.op, [A, W, Output]


@register(args=(16, 32, 32, 3, 5, 5, 10, 1, 2))
def conv2d_channel_batch(B, N, M, C, K, L, O, stride=1, padding=0, dtype="float32"):
    Apad = tvm.placeholder((B, N + 2 * padding, M + 2 * padding, C), dtype=dtype, name="Apad")
    W = tvm.placeholder((K, L, C, O), dtype=dtype, name="W")
    N_out = max(0, (N + padding * 2 - K) // stride) + 1
    M_out = max(0, (M + padding * 2 - L) // stride) + 1
    # Apad = tvm.compute((B, N + 2 * padding, M + 2 * padding, C),
    #                    lambda b, i, j, k: tvm.if_then_else(tvm.all(i >= padding, j >= padding, i < N + padding, j < M + padding),
    #                                               A[b, i - padding, j - padding, k], 0.0), name="Apad")
    rx, ry = tvm.reduce_axis((0, K), name="rx"), tvm.reduce_axis((0, L), name="ry")
    rc = tvm.reduce_axis((0, C), name="rc")
    Output = tvm.compute((B, N_out, M_out, O),
                         lambda b, i, j, k: tvm.sum(Apad[b, i * stride + rx, j * stride + ry, rc] * W[rx, ry, rc, k],
                                                    axis=[rx, ry, rc]),
                         name="Output")
    return Output.op, [Apad, W, Output]


@register(args=(32, 32, 32, 5, 5, 5, 1, 2))
def conv3d(N, M, P, K, L, Q, stride=1, padding=0, dtype="float32"):
    A = tvm.placeholder((N, M, P), dtype=dtype, name="A")
    W = tvm.placeholder((K, L, Q), dtype=dtype, name="W")
    N_out = max(0, (N + padding * 2 - K) // stride) + 1
    M_out = max(0, (M + padding * 2 - L) // stride) + 1
    P_out = max(0, (P + padding * 2 - Q) // stride) + 1
    Apad = tvm.compute((N + 2 * padding, M + 2 * padding, P + 2 * padding),
                       lambda i, j, k: tvm.if_then_else(
                           tvm.all(i >= padding, j >= padding, k >= padding, i < N + padding, j < M + padding,
                                   k < P + padding),
                           A[i - padding, j - padding, k - padding], 0.0), name="Apad")
    rx, ry, rz = tvm.reduce_axis((0, K), name="rx"), tvm.reduce_axis((0, L), name="ry"), tvm.reduce_axis((0, Q),
                                                                                                         name="rz")
    Output = tvm.compute((N_out, M_out, P_out),
                         lambda i, j, k: tvm.sum(
                             Apad[i * stride + rx, j * stride + ry, k * stride + rz] * W[rx, ry, rz],
                             axis=[rx, ry, rz]),
                         name="Output")
    return Output.op, [A, W, Output]


@register(args=(16, 32, 32, 32, 5, 5, 5, 1, 2))
def conv3d_batch(B, N, M, P, K, L, Q, stride=1, padding=0, dtype="float32"):
    A = tvm.placeholder((B, N, M, P), dtype=dtype, name="A")
    W = tvm.placeholder((K, L, Q), dtype=dtype, name="W")
    N_out = max(0, (N + padding * 2 - K) // stride) + 1
    M_out = max(0, (M + padding * 2 - L) // stride) + 1
    P_out = max(0, (P + padding * 2 - Q) // stride) + 1
    Apad = tvm.compute((B, N + 2 * padding, M + 2 * padding, P + 2 * padding),
                       lambda b, i, j, k: tvm.if_then_else(
                           tvm.all(i >= padding, j >= padding, k >= padding, i < N + padding, j < M + padding,
                                   k < P + padding),
                           A[b, i - padding, j - padding, k - padding], 0.0), name="Apad")
    rx, ry, rz = tvm.reduce_axis((0, K), name="rx"), tvm.reduce_axis((0, L), name="ry"), tvm.reduce_axis((0, Q),
                                                                                                         name="rz")
    Output = tvm.compute((B, N_out, M_out, P_out),
                         lambda b, i, j, k: tvm.sum(
                             Apad[b, i * stride + rx, j * stride + ry, k * stride + rz] * W[rx, ry, rz],
                             axis=[rx, ry, rz]),
                         name="Output")
    return Output.op, [A, W, Output]


@register(args=(32, 32, 32, 3, 5, 5, 5, 10, 1, 2))
def conv3d_channel(N, M, P, C, K, L, Q, O, stride=1, padding=0, dtype="float32"):
    A = tvm.placeholder((N, M, P, C), dtype=dtype, name="A")
    W = tvm.placeholder((K, L, Q, C, O), dtype=dtype, name="W")
    N_out = max(0, (N + padding * 2 - K) // stride) + 1
    M_out = max(0, (M + padding * 2 - L) // stride) + 1
    P_out = max(0, (P + padding * 2 - Q) // stride) + 1
    Apad = tvm.compute((N + 2 * padding, M + 2 * padding, P + 2 * padding, C),
                       lambda i, j, k, c: tvm.if_then_else(
                           tvm.all(i >= padding, j >= padding, k >= padding, i < N + padding, j < M + padding,
                                   k < P + padding),
                           A[i - padding, j - padding, k - padding, c], 0.0), name="Apad")
    rx, ry, rz, rc = tvm.reduce_axis((0, K), name="rx"), tvm.reduce_axis((0, L), name="ry"), \
                     tvm.reduce_axis((0, Q), name="rz"), tvm.reduce_axis((0, C), name="rc")
    Output = tvm.compute((N_out, M_out, P_out, O),
                         lambda i, j, k, o: tvm.sum(
                             Apad[i * stride + rx, j * stride + ry, k * stride + rz, rc] * W[rx, ry, rz, rc, o],
                             axis=[rx, ry, rz, rc]),
                         name="Output")
    return Output.op, [A, W, Output]


@register(args=(16, 32, 32, 32, 3, 5, 5, 5, 10, 2, 2))
def conv3d_channel_batch(B, N, M, P, C, K, L, Q, O, stride=1, padding=0, dtype="float32"):
    A = tvm.placeholder((B, N, M, P, C), dtype=dtype, name="A")
    W = tvm.placeholder((K, L, Q, C, O), dtype=dtype, name="W")
    N_out = max(0, (N + padding * 2 - K) // stride) + 1
    M_out = max(0, (M + padding * 2 - L) // stride) + 1
    P_out = max(0, (P + padding * 2 - Q) // stride) + 1
    Apad = tvm.compute((B, N + 2 * padding, M + 2 * padding, P + 2 * padding, C),
                       lambda b, i, j, k, c: tvm.if_then_else(
                           tvm.all(i >= padding, j >= padding, k >= padding, i < N + padding, j < M + padding,
                                   k < P + padding),
                           A[b, i - padding, j - padding, k - padding, c], 0.0), name="Apad")
    rx, ry, rz, rc = tvm.reduce_axis((0, K), name="rx"), tvm.reduce_axis((0, L), name="ry"), \
                     tvm.reduce_axis((0, Q), name="rz"), tvm.reduce_axis((0, C), name="rc")
    Output = tvm.compute((B, N_out, M_out, P_out, O),
                         lambda b, i, j, k, o: tvm.sum(
                             Apad[b, i * stride + rx, j * stride + ry, k * stride + rz, rc] * W[rx, ry, rz, rc, o],
                             axis=[rx, ry, rz, rc]),
                         name="Output")
    return Output.op, [A, W, Output]


@register(args=(102, 102))
def gaussian_blur3x3(M, N, dtype="float32"):
    A = tvm.placeholder((M, N), dtype=dtype, name="A")
    pad = (3 - 1) // 2
    Apad = tvm.compute((M + 2 * pad, N), lambda i, j: tvm.if_then_else(tvm.all(i >= pad, i < M + pad), A[i - pad, j], 0.0), name="Apad")
    B = tvm.compute((M, N), lambda i, j: (Apad[i, j] + Apad[i + 1, j] + Apad[i + 2, j]) / 3, name="B")
    Bpad = tvm.compute((M, N + 2 * pad), lambda i, j: tvm.if_then_else(tvm.all(j >= pad, j < N + pad), B[i, j - pad], 0.0), name="Bpad")
    C = tvm.compute((M, N), lambda i, j: (Bpad[i, j] + Bpad[i, j + 1] + Bpad[i, j + 2]) / 3, name="C")
    return C.op, [A, C]
