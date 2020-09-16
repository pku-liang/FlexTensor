import tvm
import numpy as np


dtype = "float32"

A = tvm.te.placeholder([4, 4], dtype=dtype, name="A")

B = tvm.te.compute([4, 4], lambda i, j: A[i, j] + 1, name="B")

C = tvm.te.compute([4, 4], lambda i, j: A[i, j] * 2, name="C")

target = "llvm"

s1 = tvm.te.create_schedule(B.op)

s2 = tvm.te.create_schedule(C.op)

s3 = tvm.te.create_schedule([B.op, C.op])

func1 = tvm.build(s1, [A, B], target=target)

func2 = tvm.build(s2, [A, C], target=target)

func3 = tvm.build(s3, [A, B, C], target=target)

ctx = tvm.context(target)

A_np = np.random.uniform(-1, 1, [4, 4]).astype(dtype)
B_np = np.zeros([4, 4]).astype(dtype)
C_np = np.zeros([4, 4]).astype(dtype)

print("Inputs:")
print(A_np)


def run(func, id):
  A_tvm = tvm.nd.array(A_np, ctx)
  B_tvm = tvm.nd.array(B_np, ctx)
  C_tvm = tvm.nd.array(C_np, ctx)
  if id == 0:
    func(A_tvm, B_tvm)
    print("Outputs:")
    print(B_tvm.asnumpy())
  elif id == 1:
    func(A_tvm, C_tvm)
    print("Outputs:")
    print(C_tvm.asnumpy())
  elif id == 2:
    func(A_tvm, B_tvm, C_tvm)
    print("Outputs 1:")
    print(B_tvm.asnumpy())
    print("Outputs 2:")
    print(C_tvm.asnumpy())

run(func1, 0)
run(func2, 1)
run(func3, 2)

