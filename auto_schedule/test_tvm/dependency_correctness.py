import tvm

A = tvm.placeholder((32, 32), dtype="float32", name="A")
B = tvm.compute((32, 30), lambda i, j: (A[i, j] + A[i, j+1] + A[i, j+2]) / 3, name="B")
C = tvm.compute((30, 30), lambda k, l: (B[k, l] + B[k + 1, l] + B[k + 2, k]), name="C")

s = tvm.create_schedule(C.op)
print(str(tvm.lower(s, [A, C], simple_mode=True)))
s[B].compute_at(s[C], s[C].op.axis[1])
print(str(tvm.lower(s, [A, C], simple_mode=True)))