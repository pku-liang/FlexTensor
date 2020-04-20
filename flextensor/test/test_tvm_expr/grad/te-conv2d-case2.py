import tvm
import numpy as np 
import torch


N = 2
nC = 16
H = 14
W = 14
K = 8
R = 3
S = 3

st = 2
group = 2

OG = K // group
IG = nC // group

P = (H - R + 1) // st
Q = (W - S + 1) // st

dtype = "float32"

A = tvm.te.placeholder([N, nC, H, W], dtype=dtype, name="A")
B = tvm.te.placeholder([K, nC, R, S], dtype=dtype, name="B")
c = tvm.te.reduce_axis([0, nC], name="c")
r = tvm.te.reduce_axis([0, R], name="r")
s = tvm.te.reduce_axis([0, S], name="s")
C = tvm.te.compute([N, K, P, Q],
  lambda n, k, h, w :
    tvm.te.sum(A[n, c, h * st + r, w * st + s] * B[k, c, r, s], axis=[c,r,s]), name="C")

dC = tvm.te.placeholder([N, K, P, Q], dtype=dtype, name="dC")

print(C.op.body)

print(dir(C.op.body[0].source[0]))

print(tvm.te.expr_equal(C.op.body[0].source[0].b.args[0], C.op.body[0].source[0].b.args[1]))

dA = tvm.te.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, B, dC, dA], simple_mode=True))

func = tvm.build(s, [A, B, dC, dA], target="llvm")

A_np = np.random.uniform(-1, 1, [N, nC, H, W]).astype("float32")
# B_np = np.ones([K, nC, R, S]).astype("float32")
B_np = np.random.uniform(-1, 1, [K, nC, R, S]).astype("float32")
# dC_np = np.ones([N, K, P, Q]).astype("float32")
dC_np = np.random.uniform(-1, 1, [N, K, P, Q]).astype("float32")
dA_np = np.zeros([N, nC, H, W]).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
B_tvm = tvm.nd.array(B_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, B_tvm, dC_tvm, dA_tvm)


# =======>
# compare the results with pytorch
A_torch = torch.tensor(A_np)
B_torch = torch.tensor(B_np)
dC_torch = torch.tensor(dC_np)
#without output_padding=1: shapes (2, 16, 14, 14), golden:(2, 16, 13, 13) mismatch
golden_torch = torch.nn.functional.conv_transpose2d(dC_torch, B_torch, stride=(st, st), output_padding=1)
# print("da_tvm", dA_tvm.shape)
# print("golden_shape,", golden_torch.size())
# print("dA_tvm:", dA_tvm)
# print("golden_torch", golden_torch)
tvm.testing.assert_allclose(dA_tvm.asnumpy(), golden_torch.numpy(), rtol=1e-3)
print("Compare with PyTorch success!")
