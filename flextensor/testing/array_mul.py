import tvm 
import numpy as np
from flextensor.utils import to_tuple


def array_mul(N):
    A = tvm.placeholder((N,), dtype="float32")
    B = tvm.placeholder((N,), dtype="float32")
    C = tvm.compute((N,), lambda i: A[i] + B[i])
    return [C.op], [A, B, C]



def test_array_mul(extent=1024, target="llvm", dev_id=0, number=10, verbose=False):
    time_cost_lst = []
    for N in range(1, extent+1):
        ctx = tvm.context(target, dev_id)
        ary_ops, ary_bufs = array_mul(N)
        ary_inputs = [tvm.nd.array(np.random.uniform(size=to_tuple(buf.shape)).astype(buf.dtype), ctx) for buf in ary_bufs[:-1]]
        ary_inputs += [tvm.nd.array(np.zeros(shape=to_tuple(buf.shape), dtype=buf.dtype), ctx) for buf in ary_bufs[-1:]]

        s = tvm.create_schedule(ary_ops)
        func = tvm.build(s, ary_bufs, target)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number)

        cost = evaluator(*ary_inputs).mean * 1e3
        # print("N=", N, "cost=", "%f(ms)"%cost, "(target=%s, dev_id=%d, number=%d)"%(target, dev_id, number))
        time_cost_lst.append(cost)
    
    res_lst = [x / time_cost_lst[0] for x in time_cost_lst]
    print("array_mul |(target=%s, dev_id=%d, number=%d)"%(target, dev_id, number))
    if verbose:
        for i, res in enumerate(res_lst):
            print("time_cost: ext=%d / ext=1 = %f"%(i + 1, res))
    else:
        print("time_cost: ext=%d / ext=1 = %f"%(extent, res_lst[-1]))


if __name__ == "__main__":
    test_array_mul(extent=1024, number=1000, verbose=True)