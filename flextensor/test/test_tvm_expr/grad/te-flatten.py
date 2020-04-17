import tvm


H = 8
W = 16

dtype = "float32"

A = tvm.te.placeholder([H, W], dtype=dtype, name="A")
C = tvm.te.compute([H * W],
  lambda i:
    A[i//W, i%W], name="C")

dC = tvm.te.compute([H * W], lambda h: 1.0, name="dC")

dA = tvm.te.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))
