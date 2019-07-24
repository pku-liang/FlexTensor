import tvm 


A = tvm.placeholder((10, 10), name="A")
B0, B1 = tvm.compute((10, 10), lambda i, j: (A[i, j] + 1, A[i, j] * 2), name="B")
s = tvm.create_schedule(B1.op)
cache0 = s.cache_write(B0.op.output(0), "local")
cache1 = s.cache_write(B1.op.output(1), "local")
print(tvm.lower(s, [A, B0, B1], simple_mode=True))
func = tvm.build(s, [A, B0, B1], "llvm")