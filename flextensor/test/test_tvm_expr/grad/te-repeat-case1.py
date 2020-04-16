import tvm


H = 8

dtype = "float32"

A = tvm.te.placeholder([H, H], dtype=dtype, name="A")
k = tvm.te.reduce_axis([0, H], name="k")
C = tvm.te.compute([H, H],
  lambda h, w :
    tvm.te.sum(A[h, k] * A[k, w], axis=[k]), name="C")

dC = tvm.te.compute([H, H], lambda h, w: 1, name="dC")

print(tvm.te.grad_op(A, C, dC))
