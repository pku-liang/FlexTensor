import tvm

A = tvm.placeholder((32, 32, 32, 32), dtype="float32", name="A")
B = tvm.placeholder((32, 32), dtype="float32", name="B")
k = tvm.reduce_axis((0, 32), name="k")
C = tvm.compute((30, 30, 32, 32), lambda a, b, c, d: tvm.sum(A[a, b, c, k] * B[k, d], axis=k), name="C")
D = tvm.compute((30, 30, 32, 32), lambda h, k, l, m: (C[h, k, l, m] * 2), name="D")
E = tvm.compute((30, 30, 32, 32), lambda h, k, l, m: (C[h, k, l, m] * 3), name="E")
F = tvm.compute((30, 30, 32, 32), lambda h, k, l, m: (D[h, k, l, m] + E[h, k, l, m]), name="F")

s = tvm.create_schedule(F.op)
s[C].compute_inline()
print(str(tvm.lower(s, [A, F], simple_mode=True)))
