import tvm

"""
We can't build a single placeholder in TVM,
which will returns Segmentation Fault
"""

A = tvm.te.placeholder([4, 4], dtype="float32", name="A")

s = tvm.te.create_schedule(A.op)

func = tvm.build(s, [A], "llvm")
