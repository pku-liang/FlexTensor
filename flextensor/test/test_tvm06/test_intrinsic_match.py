import tvm


N = 4
C = 1024
P = 14
Q = 14
K = 512
R = 3
S = 3
H = P + R//2*2
W = Q + S//2*2

dtype = "float32"


def gemm_intrinsic_compute():
  A = tvm.placeholder([32, 32], name="AA")
  B = tvm.placeholder([32, 32], name="BB")
  k = tvm.reduce_axis([0, 32], name="kk")
  Out = tvm.compute([32, 32], lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=[k]), name="OO")
  return Out


def test1():
  A = tvm.placeholder([N, H, W, C], dtype=dtype, name="A")
  Weight = tvm.placeholder([R, S, C, K], dtype=dtype, name="W")
  rc = tvm.reduce_axis([0, C], name="rc")
  rr = tvm.reduce_axis([0, R], name="rr")
  rs = tvm.reduce_axis([0, S], name="rs")
  Out = tvm.compute([N, K, P, Q],
    lambda b, k, p, q: tvm.sum(A[b, p+rr, q+rs, rc] * Weight[rr, rs, rc, k], axis=[rc, rr, rs]), name="Out")

  b, k, p, q = Out.op.axis

  intrin_t = gemm_intrinsic_compute()

  print("Target compute:")
  print(Out.op.body[0])

  print("Intrin compute:")
  print(intrin_t.op.body[0])

  print("match = ", tvm.ir_pass.intrinsic_match(Out, intrin_t, [q.var, k.var], [rc.var]))

  



if __name__ == "__main__":
  test1()