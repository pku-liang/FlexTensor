import tvm
import numpy as np 
import torch

batch_size = 3
num_classes = 5
shape_size = [batch_size, num_classes]
dtype = "float32"
ltype = "int64"

A = tvm.te.placeholder(shape_size, dtype=dtype, name="A", requires_grad=True)
targets = tvm.te.placeholder(shape_size, dtype=dtype, name="targets", requires_grad=False)
n = tvm.te.reduce_axis([0, A.shape[0]], name="n")
k = tvm.te.reduce_axis([0, A.shape[1]], name="k")
loss = tvm.te.compute([1], lambda i: tvm.te.sum(
     (A[i + n, k]-targets[n, k])*(A[i + n, k]-targets[n, k]), axis=[n, k]), name="mse", requires_grad=True)

dloss = tvm.te.placeholder([1], dtype=dtype, name="dloss")
dA, = tvm.te.mygradient(loss, [A], dloss)

s = tvm.te.create_schedule([loss.op, dA.op])

print(tvm.lower(s, [A, targets, loss, dloss, dA], simple_mode=True))

func = tvm.build(s, [A, targets, loss, dloss, dA], target="llvm")

A_np = np.random.uniform(-10, 10, shape_size).astype(dtype)
dA_np = np.zeros([batch_size, num_classes]).astype(dtype) * 0 + 1
labels_np = np.random.randint(0, num_classes, [batch_size]).astype(ltype)
targets_np = np.zeros([batch_size, num_classes]).astype(dtype)
for i in range(batch_size):
  targets_np[i][labels_np[i]] = 1.0
loss_np = np.zeros([1]).astype(dtype)
dloss_np = np.random.uniform(-1, 1, [1]).astype(dtype) * 0 + 1

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)
targets_tvm = tvm.nd.array(targets_np, ctx)
loss_tvm = tvm.nd.array(loss_np, ctx)
dloss_tvm = tvm.nd.array(dloss_np, ctx)

func(A_tvm, targets_tvm, loss_tvm, dloss_tvm, dA_tvm)

print("loss_tvm", loss_tvm)
print("dA_tvm", dA_tvm)

# =======>
# compare the results with pytorch
A_torch = torch.tensor(A_np, requires_grad=True)
targets_torch = torch.tensor(targets_np)
loss_torch = torch.nn.functional.mse_loss(A_torch, targets_torch, reduction="sum")
print("loss_pytorch", loss_torch.detach().numpy())
loss_torch.backward()
print("dA_pytorch", A_torch.grad.numpy())
tvm.testing.assert_allclose(dA_tvm.asnumpy(), A_torch.grad.numpy(), rtol=1e-30, atol=1e-30)
print("Compare to PyTorch success!")
