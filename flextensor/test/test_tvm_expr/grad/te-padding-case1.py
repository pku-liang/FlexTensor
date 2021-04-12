import tvm
import numpy as np 
import torch


N = 2
nC = 16
H = 14
W = 14
K = 16
R = 3
S = 3

padding = 1

P = H + 2 * padding
Q = W + 2 * padding

dtype = "float32"

A = tvm.te.placeholder([N, nC, H, W], dtype=dtype, name="A")
C = tvm.te.compute([N, K, P, Q],
  lambda n, k, h, w :
    tvm.tir.if_then_else(
      tvm.tir.all(h >= padding, h < P-padding, w >= padding, w < Q-padding),
      A[n, k, h-padding, w-padding], 0.0),
  name="C")

dC = tvm.te.placeholder([N, K, P, Q], dtype=dtype, name="dC")

print(C.op.body[0].name)

print(type(C.op.body[0].args[1]))

dA = tvm.te.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))

func = tvm.build(s, [A, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, [N, nC, H, W]).astype("float32")
dC_np = np.random.uniform(-10, 10, [N, K, P, Q]).astype("float32")
dA_np = np.zeros([N, nC, H, W]).astype("float32")

ctx = tvm.device("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, dC_tvm, dA_tvm)

print(dA_tvm)

# =======>
# compare the results with numpy
golden_np = dC_np[:,:, padding:P-padding, padding:Q-padding]
tvm.testing.assert_allclose(dA_tvm.asnumpy(), golden_np, rtol=1e-30)
print("Compare with Numpy success!")