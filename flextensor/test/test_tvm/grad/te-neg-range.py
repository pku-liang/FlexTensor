import tvm


N = 2
C = 16
H = 14
W = 14
K = 8
R = 3
S = 3

st = 1
group = 2

OG = K // group
IG = C // group

P = (H - R + 1) // st
Q = (W - S + 1) // st

dtype = "float32"

A = tvm.te.placeholder([N, C, H, W], dtype=dtype, name="A")
B = tvm.te.placeholder([K, IG, R, S], dtype=dtype, name="B")
c = tvm.te.reduce_axis([0, C], name="c")
r = tvm.te.reduce_axis([-R, 0], name="r")
s = tvm.te.reduce_axis([0, S], name="s")
C = tvm.te.compute([N, K, P, Q],
  lambda n, k, h, w :
    tvm.te.sum(A[n, c, h * st + (-1) * r, w * st + s] * B[k, c, (-1) * r, s], axis=[c,r,s]), name="C")

s = tvm.te.create_schedule(C.op)

print(tvm.lower(s, [A, B, C], simple_mode=True))
