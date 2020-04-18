import tvm
import numpy as np
import torch


H = 8
W = 4
K = 3

dtype = "float32"

A = tvm.te.placeholder([H, K], dtype=dtype, name="A")
B = tvm.te.placeholder([K, W], dtype=dtype, name="A")
k = tvm.te.reduce_axis([0, K], name="k")
C = tvm.te.compute([H, W],
  lambda h, w :
    tvm.te.sum(A[h, k] * B[k, w], axis=[k]), name="C")

dC = tvm.te.placeholder([H, W], dtype=dtype, name="dC")

dA = tvm.te.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, B, dC, dA], simple_mode=True))

func = tvm.build(s, [A, B, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, [H, K]).astype("float32")
B_np = np.random.uniform(-10, 10, [K, W]).astype("float32")
dC_np = np.random.uniform(-10, 10, [H, W]).astype("float32")
dA_np = np.zeros([H, K]).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
B_tvm = tvm.nd.array(B_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, B_tvm, dC_tvm, dA_tvm)

print(dA_tvm)

# =======>
# compare the results with numpy
golden_np = np.matmul(dC_np, B_np.T)
tvm.testing.assert_allclose(dA_tvm.asnumpy(), golden_np, rtol=1e-5)
print("Compare with Numpy success!")


