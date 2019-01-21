import tvm

A = tvm.placeholder((32, 32, 32, 32), dtype="float32", name="A")
B = tvm.compute((32, 30, 32, 32), lambda i, j, p, q: (A[i, j, p, q] + A[i, j+1, p, q] + A[i, j+2, p, q]) / 3, name="B")
C = tvm.compute((30, 30, 32, 32), lambda a, b, c, d: (B[a, b, c, d] + B[a + 1, b, c, d] + B[a + 2, b, c, d]), name="C")
D = tvm.compute((30, 30, 32, 32), lambda h, k, l, m: (C[h, k, l, m] * 2), name="D")

s = tvm.create_schedule(D.op)
print(str(tvm.lower(s, [A, D], simple_mode=True)))
# h, k, l, m = s[D].op.axis
# ho, hi = s[D].split(h, factor=4)
# ko, ki = s[D].split(k, factor=4)
# lo, li = s[D].split(l, factor=4)
# mo, mi = s[D].split(m, factor=4)
# s[D].reorder(ho, ko, lo, mo, hi, ki, li, mi)
# s[C].compute_at(s[D], mi)
# s[B].compute_at(s[C], s[C].op.axis[1])
s[C].compute_inline()
print(str(tvm.lower(s, [A, D], simple_mode=True)))
