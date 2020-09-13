import tvm
from flextensor.intrinsic import INTRIN_TABLE, target_embedding


def test1():
  M, K, N = 64, 64, 64
  A = tvm.placeholder([M, K], name="A", dtype="int8")
  B = tvm.placeholder([K, N], name="B", dtype="int8")
  k = tvm.reduce_axis([0, K], name="k")

  C = tvm.compute([M, N], lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=[k]), "C")
  print(INTRIN_TABLE)
  intrinsic = INTRIN_TABLE[target_embedding["c -device=micro_dev"]][0]

  output_t, inputs = intrinsic.func(*intrinsic.args)

  i, j = C.op.axis
  
  match = (tvm.ir_pass.intrinsic_match(C, output_t, [i.var, j.var], [k.var]))
  print(match)


if __name__ == "__main__":
  test1()