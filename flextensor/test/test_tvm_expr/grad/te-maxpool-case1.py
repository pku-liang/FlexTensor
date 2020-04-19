import tvm


H = 8
W = 16

R = 2
S = 2

P = H // R
Q = W // S

dtype = "float32"

A = tvm.te.placeholder([H, W], dtype=dtype, name="A")

def max_of_two(a, b):
  return (a + b - tvm.tir.if_then_else(a > b, a - b, b - a)) / 2

def max_of_any(*args):
  if (len(args) == 1):
    return args[0]
  elif (len(args) == 2):
    return max_of_two(args[0], args[1])
  else:
    return max_of_two(max_of_any(*args[:-2]), args[-1])

def helper(A, i, j, m, n):
  args = []
  for r in range(m):
    for s in range(n):
      args.append(A[i*m+r, j*n+s])
  return max_of_any(*args)

C = tvm.te.compute([P, Q],
  lambda i, j: helper(A, i, j, R, S), name="C")

dC = tvm.te.compute([P, Q], lambda h, w: 1.0, name="dC")

print(C.op.body)

dA = tvm.te.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))


"""
((((((select((((0 <= z0) && (z0 < 8)) && ((0 <= z1) && (z1 < 16))), dC(floordiv(z0, 2), floordiv(z1, 2)), 0f) + select((((0 <= z0) && (z0 < 8)) && ((1 <= z1) && (z1 < 17))), dC(floordiv(z0, 2), floordiv((z1 - 1), 2)), 0f)) - tvm_if_then_else((A((floordiv(z0, 2)*2), ((floordiv(z1, 2)*2) - 1)) < A((floordiv(z0, 2)*2), ((floordiv(z1, 2)*2) - 2))), (select((((0 <= z0) && (z0 < 8)) && ((1 <= z1) && (z1 < 17))), dC(floordiv(z0, 2), floordiv((z1 - 1), 2)), 0f) - select((((0 <= z0) && (z0 < 8)) && ((2 <= z1) && (z1 < 18))), dC(floordiv(z0, 2), (floordiv(z1, 2) - 1)), 0f)), select((((0 <= z0) && (z0 < 8)) && ((3 <= z1) && (z1 < 19))), (dC(floordiv(z0, 2), floordiv((z1 - 3), 2)) - dC(floordiv(z0, 2), floordiv((z1 - 3), 2))), 0f)))*0.5f) + select((((1 <= z0) && (z0 < 9)) && ((4 <= z1) && (z1 < 20))), dC(floordiv((z0 - 1), 2), (floordiv(z1, 2) - 2)), 0f)) - tvm_if_then_else((A(((floordiv(z0, 2)*2) - 1), ((floordiv(z1, 2)*2) - 7)) < (((A(((floordiv(z0, 2)*2) - 2), ((floordiv(z1, 2)*2) - 8)) + A(((floordiv(z0, 2)*2) - 2), ((floordiv(z1, 2)*2) - 7))) - tvm_if_then_else((A(((floordiv(z0, 2)*2) - 2), ((floordiv(z1, 2)*2) - 7)) < A(((floordiv(z0, 2)*2) - 2), ((floordiv(z1, 2)*2) - 8))), (A(((floordiv(z0, 2)*2) - 2), ((floordiv(z1, 2)*2) - 8)) - A(((floordiv(z0, 2)*2) - 2), ((floordiv(z1, 2)*2) - 7))), (A(((floordiv(z0, 2)*2) - 2), ((floordiv(z1, 2)*2) - 7)) - A(((floordiv(z0, 2)*2) - 2), ((floordiv(z1, 2)*2) - 8)))))*0.5f)), ((((select((((1 <= z0) && (z0 < 9)) && ((4 <= z1) && (z1 < 20))), dC(floordiv((z0 - 1), 2), (floordiv(z1, 2) - 2)), 0f) + select((((1 <= z0) && (z0 < 9)) && ((5 <= z1) && (z1 < 21))), dC(floordiv((z0 - 1), 2), floordiv((z1 - 5), 2)), 0f)) - tvm_if_then_else((A((floordiv((z0 - 1), 2)*2), ((floordiv(z1, 2)*2) - 5)) < A((floordiv((z0 - 1), 2)*2), ((floordiv(z1, 2)*2) - 6))), (select((((1 <= z0) && (z0 < 9)) && ((5 <= z1) && (z1 < 21))), dC(floordiv((z0 - 1), 2), floordiv((z1 - 5), 2)), 0f) - select((((1 <= z0) && (z0 < 9)) && ((6 <= z1) && (z1 < 22))), dC(floordiv((z0 - 1), 2), (floordiv(z1, 2) - 3)), 0f)), select((((1 <= z0) && (z0 < 9)) && ((7 <= z1) && (z1 < 23))), (dC(floordiv((z0 - 1), 2), floordiv((z1 - 7), 2)) - dC(floordiv((z0 - 1), 2), floordiv((z1 - 7), 2))), 0f)))*0.5f) - select((((2 <= z0) && (z0 < 10)) && ((8 <= z1) && (z1 < 24))), dC((floordiv(z0, 2) - 1), (floordiv(z1, 2) - 4)), 0f)), (select((((3 <= z0) && (z0 < 11)) && ((9 <= z1) && (z1 < 25))), dC(floordiv((z0 - 3), 2), floordiv((z1 - 9), 2)), 0f) - (((select((((3 <= z0) && (z0 < 11)) && ((9 <= z1) && (z1 < 25))), dC(floordiv((z0 - 3), 2), floordiv((z1 - 9), 2)), 0f) + select((((3 <= z0) && (z0 < 11)) && ((10 <= z1) && (z1 < 26))), dC(floordiv((z0 - 3), 2), (floordiv(z1, 2) - 5)), 0f)) - tvm_if_then_else((A((floordiv((z0 - 3), 2)*2), ((floordiv((z1 - 11), 2)*2) + 1)) < A((floordiv((z0 - 3), 2)*2), (floordiv((z1 - 11), 2)*2))), (select((((3 <= z0) && (z0 < 11)) && ((10 <= z1) && (z1 < 26))), dC(floordiv((z0 - 3), 2), (floordiv(z1, 2) - 5)), 0f) - select((((3 <= z0) && (z0 < 11)) && ((11 <= z1) && (z1 < 27))), dC(floordiv((z0 - 3), 2), floordiv((z1 - 11), 2)), 0f)), select((((3 <= z0) && (z0 < 11)) && ((12 <= z1) && (z1 < 28))), (dC(floordiv((z0 - 3), 2), (floordiv(z1, 2) - 6)) - dC(floordiv((z0 - 3), 2), (floordiv(z1, 2) - 6))), 0f)))*0.5f))))*0.5f)
"""