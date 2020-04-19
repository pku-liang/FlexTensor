import tvm
import numpy as np 
import torch


N = 2
H = 14

dtype = "float32"

def softmax(inputs):
  K = inputs.shape[-1]
  k = tvm.te.reduce_axis([0, K], name="k")
  exp_val = tvm.te.compute(inputs.shape, lambda n, h: tvm.tir.exp(inputs[n, h]), name="Softmax_exp")
  sum_val = tvm.te.compute(exp_val.shape[:-1], lambda n: tvm.te.sum(exp_val[n, k], axis=[k]), name="Softmax_sum")
  final_val = tvm.te.compute(exp_val.shape, lambda n, h: exp_val[n, h]/sum_val[n], name="Softmax_div")
  return [exp_val, sum_val, final_val]

A = tvm.te.placeholder([N, H], dtype=dtype, name="A")
B, C, D = softmax(A)

dD = tvm.te.placeholder([N, H], dtype=dtype, name="dC")

dA, = tvm.te.mygradient(D, [A])

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dD, dA], simple_mode=True))

func = tvm.build(s, [A, dD, dA], target="llvm")

# A_np = np.random.uniform(-10, 10, [N, nC, H, W]).astype("float32")
# B_np = np.random.uniform(-10, 10, [K, nC, R, S]).astype("float32")
# dC_np = np.random.uniform(-10, 10, [N, K, P, Q]).astype("float32")
# dA_np = np.zeros([N, nC, H, W]).astype("float32")

# ctx = tvm.context("llvm", 0)
# A_tvm = tvm.nd.array(A_np, ctx)
# B_tvm = tvm.nd.array(B_np, ctx)
# dC_tvm = tvm.nd.array(dC_np, ctx)
# dA_tvm = tvm.nd.array(dA_np, ctx)

# func(A_tvm, B_tvm, dC_tvm, dA_tvm)

# print(dA_tvm)

# # =======>
# # compare the results with pytorch
# A_torch = torch.tensor(A_np)
# B_torch = torch.tensor(B_np)
# dC_torch = torch.tensor(dC_np)
# golden_torch = torch.nn.functional.conv_transpose2d(dC_torch, B_torch)
# tvm.testing.assert_allclose(dA_tvm.asnumpy(), golden_torch.numpy(), rtol=1e-3)
# print("Compare with PyTorch success!")