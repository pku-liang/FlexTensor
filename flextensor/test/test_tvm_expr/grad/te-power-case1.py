import tvm
import numpy as np


H = 4
W = 2

dtype = "float32"

A = tvm.te.placeholder([H, W], dtype=dtype, name="A")

C = tvm.te.compute([H, W],
  lambda h, w :
    tvm.tir.power(A[h, w]+1e-9, 2), name="C")

dC = tvm.te.placeholder([H, W], dtype=dtype, name="dC")

dA = tvm.te.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

s = tvm.te.create_schedule([C.op, dA.op])

print(tvm.lower(s, [A, C, dC, dA], simple_mode=True))

func = tvm.build(s, [A, C, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, [H, W]).astype("float32")
dC_np = np.random.uniform(-10, 10, [H, W]).astype("float32")
C_np = np.zeros([H, W]).astype("float32")
dA_np = np.zeros([H, W]).astype("float32")

ctx = tvm.device("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
C_tvm = tvm.nd.array(C_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, C_tvm, dC_tvm, dA_tvm)

print(A_tvm)
print(C_tvm)
print(dC_tvm)
print(dA_tvm)
