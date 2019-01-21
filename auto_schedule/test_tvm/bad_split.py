import tvm

A = tvm.placeholder((1024, 1024), name="A")
B = tvm.compute((1024, 1024), lambda i, j: A[i, j] * A[i, j], name="B")

s = tvm.create_schedule(B.op)
io, ii = s[B].split(s[B].op.axis[0], nparts=16)
io, ii = s[B].split(io, nparts=32)
io, ii = s[B].split(io, nparts=4)

print(tvm.lower(s, [A, B], simple_mode=True))
func = tvm.build(s, [A, B], "llvm")