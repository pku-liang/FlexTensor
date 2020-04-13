import tvm


H = 8

dtype = "float32"

A = tvm.te.placeholder([H, H], dtype=dtype, name="A")
C = tvm.te.compute([H],
  lambda i:
    A[i, i], name="C")

dC = tvm.te.compute([H], lambda h: 1, name="dC")

print(tvm.te.grad_op(A, C, dC))
