import tvm
from Environment import Env
from Extractor import CNNExtractor


N = 1024
M = 1024
L = 1024
dtype = "float32"

A = tvm.placeholder((N, M), dtype=dtype, name="A")
B = tvm.placeholder((M, L), dtype=dtype, name="B")
k = tvm.reduce_axis((0, M), name="k")
C = tvm.compute((N, L), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name="C")

env = Env(C.op)
env.print_()
node = env.op2node[C.op]
print(node.get_axis_dom("i"))

touch = CNNExtractor(env)
print(touch.extract_whole_graph_features())