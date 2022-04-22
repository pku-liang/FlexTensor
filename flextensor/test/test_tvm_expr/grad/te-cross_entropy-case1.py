import tvm 
import numpy as np
import torch


def cross_entropy(inputs, targets, weights, reduction="mean"):
  N, C = inputs.shape
  c = tvm.te.reduce_axis([0, C], "c")
  sum_val = tvm.te.compute([N], lambda i: tvm.te.sum(tvm.tir.exp(inputs[i, c]), axis=[c]), "sum_val")
  if reduction == "mean":
    rn = tvm.te.reduce_axis([0, N], "rn")
    rc = tvm.te.reduce_axis([0, C], "rc")
    sum_weight = tvm.te.compute([1], lambda i: tvm.te.sum(weights[i+rc]*targets[rn, rc]/N, axis=[rn, rc]), "mean_weight", requires_grad=False)
  elif reduction == "sum":
    sum_weight = tvm.te.compute([1], lambda i: tvm.tir.expr.const(1, weights.dtype)/N, "sum_weight", requires_grad=False)
  else:
    raise NameError()
  rrn = tvm.te.reduce_axis([0, N], "rrn")
  rrc = tvm.te.reduce_axis([0, C], "rrc")
  # return tvm.te.compute([1],
  #   lambda i: tvm.te.sum(
  #     weights[rrc] * targets[i+rrn, rrc] * (tvm.tir.log(sum_val[i+rrn]) - inputs[i+rrn, rrc]*targets[rrn, rrc])/(N*sum_weight[i]),
  #     axis=[rrn, rrc]),
  #   "cross_entropy")
  return tvm.te.compute([1],
    lambda i: tvm.te.sum(
      targets[i+rrn, rrc] * (tvm.tir.log(sum_val[i+rrn]) - inputs[i+rrn, rrc]*targets[i+rrn, rrc])/(N),
      axis=[rrn, rrc]),
    "cross_entropy")


N = 100
C = 100
dtype = "float32"
ltype = "int64"
target = "llvm"

A = tvm.te.placeholder([N, C], dtype=dtype, name="A")
targets = tvm.te.placeholder([N, C], dtype=dtype, name="targets", requires_grad=False)
labels = tvm.te.placeholder([N], dtype=ltype, name="labels", requires_grad=False)
weights = tvm.te.placeholder([C], dtype=dtype, name="weights", requires_grad=False)

loss = cross_entropy(A, targets, weights, reduction="mean")

dloss = tvm.te.placeholder([1], dtype=dtype, name="dloss")

dA, = tvm.te.mygradient(loss, [A], dloss)

s = tvm.te.create_schedule([loss.op, dA.op])

print(tvm.lower(s, [A, targets, weights, loss, dloss, dA], simple_mode=True))

func = tvm.build(s, [A, targets, weights, loss, dloss, dA], target)

A_np = np.random.uniform(-1, 1, [N, C]).astype(dtype)
dA_np = np.zeros([N, C]).astype(dtype) * 0 + 1
labels_np = np.random.randint(0, C, [N]).astype(ltype)
# labels_np[0] = 1
targets_np = np.zeros([N, C]).astype(dtype)
for i in range(N):
  targets_np[i][labels_np[i]] = 1.0
weights_np = np.random.uniform(-1, 1, [C]).astype(dtype) * 0 + 1
loss_np = np.zeros([1]).astype(dtype)
dloss_np = np.random.uniform(-1, 1, [1]).astype(dtype) * 0 + 1

ctx = tvm.device(target, 0)
A_tvm = tvm.nd.array(A_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)
targets_tvm = tvm.nd.array(targets_np, ctx)
weights_tvm = tvm.nd.array(weights_np, ctx)
loss_tvm = tvm.nd.array(loss_np, ctx)
dloss_tvm = tvm.nd.array(dloss_np, ctx)

func(A_tvm, targets_tvm, weights_tvm, loss_tvm, dloss_tvm, dA_tvm)
print(loss_tvm)
print(dA_tvm)

A_torch = torch.tensor(A_np, requires_grad=True)
labels_torch = torch.tensor(labels_np)
weights_torch = torch.tensor(weights_np)
loss_torch = torch.nn.functional.cross_entropy(A_torch, labels_torch, weights_torch, reduction="mean")

print(loss_torch.detach().numpy())
loss_torch.backward()
print(A_torch.grad.numpy())

tvm.testing.assert_allclose(dA_tvm.asnumpy(), A_torch.grad.numpy(), rtol=1e-30, atol=1e-9)
print("Compare to PyTorch success!")