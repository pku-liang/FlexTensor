import tvm


def compute_inline_reduce():
    A = tvm.placeholder((32, 32, 32, 32), dtype="float32", name="A")
    B = tvm.placeholder((32, 32), dtype="float32", name="B")
    k = tvm.reduce_axis((0, 32), name="k")
    C = tvm.compute((30, 30, 32, 32), lambda a, b, c, d: tvm.sum(A[a, b, c, k] * B[k, d], axis=k), name="C")
    D = tvm.compute((30, 30, 32, 32), lambda h, k, l, m: (C[h, k, l, m] * 2), name="D")
    E = tvm.compute((30, 30, 32, 32), lambda h, k, l, m: (C[h, k, l, m] * 3), name="E")
    F = tvm.compute((30, 30, 32, 32), lambda h, k, l, m: (D[h, k, l, m] + E[h, k, l, m]), name="F")

    s = tvm.create_schedule(F.op)
    s[C].compute_inline()
    try:
        tvm.build(s, [A, F], "llvm")
    except Exception as e:
        return False, str(e)
    return True, "pass"


def compute_inline_output():
    N = 1024
    M = 512
    A = tvm.placeholder((M, N), name="A")
    B, C = tvm.compute((M, N), lambda i, j: (A[i, j] + 1, A[i, j] * 2), name="B_C")
    D = tvm.compute((M, N), lambda i, j: B[i, j] * 2, name="D")

    s = tvm.create_schedule(D.op)
    s[B].compute_inline()
    try:
        tvm.build(s, [A, C, D], "llvm")
    except Exception as e:
        return False, str(e)
    return True, "pass"


if __name__ == "__main__":
    print("compute_inline_reduce:", *compute_inline_reduce())
    print("compute_inline_output:", *compute_inline_output())
