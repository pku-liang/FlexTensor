from __future__ import absolute_import 

import sys
import tvm 
from flextensor.task import TASK_TABLE


def print_source(s, bufs, target, file=sys.stdout):
    func = tvm.build(s, bufs, target)
    if target in ["cuda", "opencl"]:
        print(func.imported_modules[0].get_source(), file=file)

def recursive_fuse(s, cur, flag=False):
    for t in s[cur].op.input_tensors:
        if isinstance(t.op, tvm.te.tensor.ComputeOp):
            recursive_fuse(s, t.op, True)
    if flag:
        s[cur].compute_inline()

hit_set = set()
for task in TASK_TABLE.values():
    if (task.target == "cuda" and "gemm_conv" not in task.key 
        and "mttkrp" not in task.key and "block_circulant_matrix" not in task.key
        and "pixel" not in task.key and "unpool" not in task.key and "shift" not in task.key
        and "packed" not in task.key):
        prefix = task.key.rsplit("_", 4)[0]
        if prefix in hit_set:
            continue
        hit_set.add(prefix)
        outops, bufs = task.func(*task.args)
        s = tvm.te.create_schedule(outops)
        bx = tvm.te.thread_axis("blockIdx.x")
        op = outops[0]
        recursive_fuse(s, op)
        outer, inner = s[op].split(s[op].op.axis[0], nparts=1)
        s[op].bind(outer, bx)
        print(task.key)
        print_source(s, bufs, "opencl")