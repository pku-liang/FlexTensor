import topi
import tvm
import numpy as np 
import torch


dim0 = 8
dim1 = 3
dim2 = 4
shape_size1 = [dim0, dim1]
shape_size2 = [dim0, dim2]
dtype = "float32"

A = tvm.te.placeholder(shape_size1, dtype=dtype, name="A")
B = tvm.te.placeholder(shape_size2, dtype=dtype, name="B")
C = topi.concatenate([A, B], axis=1)

dC = tvm.te.placeholder(C.shape, dtype=dtype, name="dC")
dA, dB = tvm.te.mygradient(C, [A, B], dC)

s = tvm.te.create_schedule([C.op, dA.op, dB.op])

print(tvm.lower(s, [A, B, dC, dA, dB], simple_mode=True))

func = tvm.build(s, [A, B, dC, dA, dB], target="llvm")

A_np = np.random.uniform(-10, 10, shape_size1).astype("float32")
B_np = np.random.uniform(-10, 10, shape_size2).astype("float32")

dC_np = np.ones([dim0, dim1+dim2]).astype("float32") 
dA_np = np.zeros(shape_size1).astype("float32")
dB_np = np.zeros(shape_size2).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
B_tvm = tvm.nd.array(B_np, ctx)

dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)
dB_tvm = tvm.nd.array(dB_np, ctx)
func(A_tvm, B_tvm, dC_tvm, dA_tvm, dB_tvm)

print("dA_tvm", dA_tvm)

# =======>
# compare the results with pytorch
A_torch = torch.tensor(A_np, requires_grad=True)
B_torch = torch.tensor(B_np, requires_grad=True)
C_torch = torch.cat([A_torch, B_torch], dim=1)
loss = C_torch.sum()
loss.backward()
print("Pytorch gradient:\n", A_torch.grad.numpy(), B_torch.grad.numpy())
tvm.testing.assert_allclose(dA_tvm.asnumpy(), A_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
tvm.testing.assert_allclose(dB_tvm.asnumpy(), B_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
print("Compare with PyTorch success!")
