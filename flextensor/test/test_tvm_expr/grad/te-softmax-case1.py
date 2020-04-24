import tvm
import numpy as np 
import torch


N = 2
H = 14

dtype = "float32"

factor = 1


def softmax(inputs):
  K = inputs.shape[-1]
  k = tvm.te.reduce_axis([0, K], name="k")
  k1 = tvm.te.reduce_axis([0, K], name="k1")
  max_val = tvm.te.compute([N, K], lambda n, h: tvm.te.max(inputs[n, k1], axis=[k1]), name="mean_val", requires_grad=True)
  exp_val = tvm.te.compute(inputs.shape, lambda n, h: tvm.tir.exp(inputs[n, h]-max_val[n, h]), name="Softmax_exp", requires_grad=True)
  sum_val = tvm.te.compute(exp_val.shape, lambda n, h: tvm.te.sum(exp_val[n, k], axis=[k]), name="Softmax_sum", requires_grad=True)
  final_val = tvm.te.compute(exp_val.shape, lambda n, h: exp_val[n, h]/(sum_val[n, h]), name="Softmax_div", requires_grad=True)
  return [exp_val, sum_val, final_val]


def mse_loss(inputs, targets):
  N = inputs.shape[0]
  K = inputs.shape[1]
  n = tvm.te.reduce_axis([0, inputs.shape[0]], name="n")
  k = tvm.te.reduce_axis([0, inputs.shape[1]], name="k")
  # return tvm.te.compute([1], lambda i: tvm.te.sum((inputs[i + n, k]-targets[i + n, k])*(inputs[i + n, k]-targets[i + n, k])/(N*K), axis=[n, k]), name="mse")
  return tvm.te.compute([1], lambda i: tvm.te.sum(tvm.tir.power((inputs[i + n, k]-targets[i + n, k]), 2)/(N*K), axis=[n, k]), name="mse", requires_grad=True)


A = tvm.te.placeholder([N, H], dtype=dtype, name="A", requires_grad=True)
label = tvm.te.placeholder([N, H], dtype=dtype, name="label", requires_grad=False)
B, C, D = softmax(A)
E = mse_loss(D, label)
print(E.requires_grad)

dD = tvm.te.placeholder([N, H], dtype=dtype, name="dD")
dE = tvm.te.placeholder([1], dtype=dtype, name="dE")

dA, = tvm.te.mygradient(E, [A])

s = tvm.te.create_schedule([E.op, dA.op])

print(tvm.lower(s, [A, label, D, E, dA], simple_mode=True))

func = tvm.build(s, [A, label, D, E, dA], target="llvm")

A_np = np.random.uniform(-100, 100, [N, H]).astype("float32")
label_np = np.random.uniform(-1, 1, [N, H]).astype("float32")
D_np = np.zeros([N, H]).astype("float32")
E_np = np.zeros([1]).astype("float32")

dA_np = np.zeros([N, H]).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
label_tvm = tvm.nd.array(label_np, ctx)
D_tvm = tvm.nd.array(D_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)
E_tvm = tvm.nd.array(E_np, ctx)

func(A_tvm, label_tvm, D_tvm, E_tvm, dA_tvm)

# print("TVM result:\n", D_tvm)
print("TVM gradient:\n", dA_tvm)

# =======>
# compare the results with pytorch
A_torch = torch.tensor(A_np, requires_grad=True)
label_torch = torch.tensor(label_np, requires_grad=False)
E_torch = torch.tensor(E_np)

D_torch = torch.nn.functional.softmax(A_torch, dim=1)
E_torch = torch.nn.functional.mse_loss(D_torch, label_torch)
# print("Pytorch result:\n", D_torch.detach().numpy())
E_torch.backward()
print("Pytorch gradient:\n", A_torch.grad.numpy())
tvm.testing.assert_allclose(D_tvm.asnumpy(), D_torch.detach().numpy(), atol=1e-6*factor, rtol=1e-5)
tvm.testing.assert_allclose(dA_tvm.asnumpy(), A_torch.grad.numpy(), atol=1e-6*factor, rtol=1e-5)
print("Compare with PyTorch success!")
