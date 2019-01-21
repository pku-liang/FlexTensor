import tvm
import numpy as np
from auto_schedule import auto_schedule, base_line
from auto_schedule.training_examples import matmul


def evaluate(func, A, B, C, ctx, number=1):
    evaluator = func.time_evaluator(func.entry_name, ctx, number)
    A = tvm.nd.array(A, ctx)
    B = tvm.nd.array(B, ctx)
    C = tvm.nd.array(C, ctx)
    print("    cost time {}ms".format(evaluator(A, B, C).mean * 1e3))


N = 1024
M = 1024
L = 1024
args = (matmul, (N, M, L), "llvm")

baseline = base_line(*args)
func = auto_schedule(*args)

A = np.random.uniform(size=(N, M)).astype("float32")
B = np.random.uniform(size=(M, L)).astype("float32")
C = np.zeros(shape=(N, L), dtype=np.float32)

ctx = tvm.context("llvm", 0)

print("baseline:")
evaluate(baseline, A, B, C, ctx)
print("func:")
evaluate(func, A, B, C, ctx)



