import tvm


H = 8
W = 9

dtype = "float32"

A = tvm.te.placeholder([H], dtype=dtype, name="A")

C = tvm.te.compute([H, W],
  lambda h, w :
    A[h] * 4 - A[h] * A[h], name="C")

dC = tvm.te.compute([H, W], lambda h, w: 1.0, name="dC")

dA = tvm.te.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))
