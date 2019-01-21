import tvm

A = tvm.placeholder((32, 32, 32, 32), dtype="float32", name="A")
B = tvm.compute((32, 30, 32, 32), lambda i, j, p, q: (A[i, j, p, q] + A[i, j+1, p, q] + A[i, j+2, p, q]) / 3, name="B")
C = tvm.compute((30, 30, 32, 32), lambda k, l, p, q: (B[k, l, p, q] + B[k + 1, l, p, q] + B[k + 2, k, p, q]), name="C")

s = tvm.create_schedule(C.op)
print(str(tvm.lower(s, [A, C], simple_mode=True)))
s[B].compute_at(s[C], s[C].op.axis[1])
print(s[B].op.axis)
s[B].split(s[B].op.axis[2], factor=4)
print(str(tvm.lower(s, [A, C], simple_mode=True)))
