import tvm
import numpy as np


H = 6
W = 6

R = 2
S = 2

P = H // R
Q = W // S

dtype = "float32"

A = tvm.te.placeholder([H, W], dtype=dtype, name="A")

r = tvm.te.reduce_axis([0, R], name="r")
s = tvm.te.reduce_axis([0, S], name="s")

C = tvm.te.compute([P, Q],
  lambda i, j: tvm.te.sum(A[i * R + r, j * S + s]/(R*S), axis=[r, s]), name="C")

dC = tvm.te.placeholder([P, Q], dtype=dtype, name="dC")


dA = tvm.te.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))

func = tvm.build(s, [A, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, [H, W]).astype("float32")
dC_np = np.random.uniform(-10, 10, [P, Q]).astype("float32")
dA_np = np.zeros([H, W]).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, dC_tvm, dA_tvm)

print(dC_tvm)
print(dA_tvm.asnumpy())

# =======>
# compare the results with numpy
golden_np = np.zeros([H, W]).astype("float32")
for i in range(0, P):
    for j in range(0, Q):
        for di in range(0, R):
            for dj in range(0, S):
                assert(i+di < H)
                assert(j+dj < W)
                golden_np[i*R+di][j*S+dj] = dC_np[i][j] / (R * S)
tvm.testing.assert_allclose(dA_tvm.asnumpy(), golden_np, rtol=1e-30)
print("Compare with Numpy success!")
