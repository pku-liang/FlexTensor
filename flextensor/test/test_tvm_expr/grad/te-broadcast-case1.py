import tvm


H = 8
W = 16

dtype = "float32"

A = tvm.te.placeholder([H], dtype=dtype, name="A")
C = tvm.te.compute([H, W],
  lambda i, j:
    A[i], name="C")

dC = tvm.te.compute([H, W], lambda h, w: 1, name="dC")

print(tvm.te.grad_op(A, C, dC))
