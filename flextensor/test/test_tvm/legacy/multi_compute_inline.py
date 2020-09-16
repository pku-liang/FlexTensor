import tvm

A = tvm.te.placeholder((32, 32, 32, 32), dtype="float32", name="A")
B = tvm.te.compute((32, 30, 32, 32), lambda i, j, p, q: (A[i, j, p, q] + A[i, j+1, p, q] + A[i, j+2, p, q]) / 3, name="B")
C = tvm.te.compute((30, 30, 32, 32), lambda a, b, c, d: (B[a, b, c, d] + B[a + 1, b, c, d] + B[a + 2, b, c, d]), name="C")
D = tvm.te.compute((30, 30, 32, 32), lambda h, k, l, m: (C[h, k, l, m] * 2), name="D")
E = tvm.te.compute((30, 30, 32, 32), lambda h, k, l, m: (C[h, k, l, m] * 3), name="E")
F = tvm.te.compute((30, 30, 32, 32), lambda h, k, l, m: (D[h, k, l, m] + E[h, k, l, m]), name="F")

s = tvm.te.create_schedule(F.op)
s[C].compute_inline()
print(str(tvm.lower(s, [A, F], simple_mode=True)))
