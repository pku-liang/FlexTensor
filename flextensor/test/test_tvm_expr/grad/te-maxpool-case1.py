import tvm
import numpy as np 
import torch


H = 4
W = 4

R = 2
S = 2

P = H // R
Q = W // S

dtype = "float32"


def mse_loss(inputs, targets):
  N = inputs.shape[0]
  K = inputs.shape[1]
  n = tvm.te.reduce_axis([0, inputs.shape[0]], name="n")
  k = tvm.te.reduce_axis([0, inputs.shape[1]], name="k")
  # return tvm.te.compute([1], lambda i: tvm.te.sum((inputs[i + n, k]-targets[i + n, k])*(inputs[i + n, k]-targets[i + n, k])/(N*K), axis=[n, k]), name="mse")
  return tvm.te.compute([1], lambda i: tvm.te.sum(tvm.tir.power((inputs[i + n, k]-targets[i + n, k]), 2)/(N*K), axis=[n, k]), name="mse", requires_grad=True)


A = tvm.te.placeholder([H, W], dtype=dtype, name="A", requires_grad=True)
label = tvm.te.placeholder([P, Q], dtype=dtype, name="label", requires_grad=False)

p = tvm.te.reduce_axis([0, R], "p")
q = tvm.te.reduce_axis([0, S], "q")

B = tvm.te.compute([P, Q], lambda a, b: tvm.te.max(A[a*R+p, b*S+q]-1e-5, axis=[p, q]), requires_grad=False, name="max_value")
C = tvm.te.compute([H, W], lambda u, v: tvm.tir.if_then_else(A[u, v] > B[u//R, v//S], 1.0, 0.0), requires_grad=False, name="map")

r = tvm.te.reduce_axis([0, R], "r")
s = tvm.te.reduce_axis([0, S], "s")

D = tvm.te.compute([P, Q],
  lambda i, j: tvm.te.sum(A[i*R+r, j*S+s] * C[i*R+r, j*S+s], axis=[r, s]), name="D", requires_grad=True)

E = mse_loss(D, label)

dA, = tvm.te.mygradient(E, [A])

s = tvm.te.create_schedule([E.op, dA.op])

print(tvm.lower(s, [A, label, E, dA], simple_mode=True))

func = tvm.build(s, [A, label, E, dA], target="llvm")

A_np = np.random.uniform(-10, 10, [H, W]).astype("float32")
label_np = np.random.uniform(-10, 10, [P, Q]).astype("float32")
E_np = np.zeros([1]).astype("float32")
dA_np = np.zeros([H, W]).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
label_tvm = tvm.nd.array(label_np, ctx)
E_tvm = tvm.nd.array(E_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, label_tvm, E_tvm, dA_tvm)

print(E_tvm)

# ==> compare to pytorch

A0_torch = torch.tensor(A_np, requires_grad=True)
A_torch = A0_torch.unsqueeze(0).unsqueeze(1)
label_torch = torch.tensor(label_np, requires_grad=False)
D_torch = torch.max_pool2d(A_torch, [R, S]).squeeze()
E_torch = torch.nn.functional.mse_loss(D_torch, label_torch)
print(E_torch.detach().numpy())
E_torch.backward()

tvm.testing.assert_allclose(E_tvm.asnumpy(), E_torch.detach().numpy(), atol=1e-5, rtol=1e-30)
tvm.testing.assert_allclose(dA_tvm.asnumpy(), A0_torch.grad.numpy(), atol=1e-5, rtol=1e-30)