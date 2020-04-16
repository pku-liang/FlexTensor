import tvm


N = 2
C = 16
H = 14
W = 14
K = 8
R = 3
S = 3

st = 2
group = 2

OG = K // group
IG = C // group

P = (H - R + 1) // st
Q = (W - S + 1) // st

dtype = "float32"

A = tvm.te.placeholder([N, C, H, W], dtype=dtype, name="A")
B = tvm.te.placeholder([K, C, R, S], dtype=dtype, name="B")
c = tvm.te.reduce_axis([0, C], name="c")
r = tvm.te.reduce_axis([0, R], name="r")
s = tvm.te.reduce_axis([0, S], name="s")
C = tvm.te.compute([N, K, P, Q],
  lambda n, k, h, w :
    tvm.te.sum(A[n, c, h * st + r, w * st + s] * B[k, c, r, s], axis=[c,r,s]), name="C")

dC = tvm.te.compute([N, K, P, Q], lambda a, b, c, d: 1, name="dC")

print(C.op.body)

print(dir(C.op.body[0].source[0]))

print(tvm.te.expr_equal(C.op.body[0].source[0].b.args[0], C.op.body[0].source[0].b.args[1]))

print(tvm.te.grad_op(A, C, dC))
