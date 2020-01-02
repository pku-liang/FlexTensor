import tvm 
import numpy as np


def to_int(expr):
    res = int(expr)
    return res


def to_tuple(expr_tuple):
    return tuple([to_int(x) for x in expr_tuple])


# define compute

def conv2d_1x1(N, C, H, W, K):
    A = tvm.placeholder((N, C, H, W), dtype="float32", name="A")
    Weight = tvm.placeholder((K, C), dtype="float32", name="W")
    rc = tvm.reduce_axis((0, C), name="rc")
    B = tvm.compute((N, K, H, W), lambda b, k, i, j: tvm.sum(A[b, rc, i, j] * Weight[k, rc], axis=rc))
    return [B.op], [A, Weight, B]


def jlc_schedule(ops):
    blockdimx = 16
    blockdimy = 8
    threaddimy = 8
    step = 16
    output = ops[0]
    B = output.output(0)
    A = B.op.input_tensors[0]
    W = B.op.input_tensors[1]
    s = tvm.create_schedule(output)
    AA = s.cache_read(A, 'shared', [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BL = s.cache_write(B, "local")

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, blockdimx), "threadIdx.x")
    thread_y = tvm.thread_axis((0, blockdimy), "threadIdx.y")
    thread_xz = tvm.thread_axis("vthread")
    thread_yz = tvm.thread_axis("vthread")


    ni, fi, hi, wi = s[B].op.axis

    hw = s[B].fuse(hi, wi)

    by, fi = s[B].split(fi, factor=threaddimy * blockdimy)

    # Bind the iteration variables to GPU thread indices
    s[B].bind(ni, block_z)
    s[B].bind(by, block_y)
    #s[B].bind(hwo, block_x)

    fi, ty = s[B].split(fi, factor=blockdimy)
    ni, tx = s[B].split(hw, factor=blockdimx)
    #s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

    s[B].bind(fi, thread_yz)
    s[B].bind(ni, thread_xz)
    s[B].bind(ty, thread_y)
    s[B].bind(tx, thread_x)


    # Schedule BL local write
    s[BL].compute_at(s[B], tx)
    ni, fi, yi, xi = s[BL].op.axis
    rc, = s[BL].op.reduce_axis
    rco, rci = s[BL].split(rc, factor=step)

    # Attach computation to iteration variables
    s[AA].compute_at(s[BL], rco)
    s[WW].compute_at(s[BL], rco)
    s[AL].compute_at(s[BL], rci)
    s[WL].compute_at(s[BL], rci)

    # Schedule for A's shared memory load
    yi, xi, ci, ni = s[AA].op.axis
    ty, ci = s[AA].split(ci, nparts=blockdimy)
    tx, ni = s[AA].split(ni, nparts=blockdimx)
    s[AA].reorder(ty, tx, yi, xi, ci, ni)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    #s[AA].vectorize(ni)  # vectorize memory load

    # Schedule for W's shared memory load
    ci, fi= s[WW].op.axis
    ty, ci = s[WW].split(ci, nparts=blockdimy)
    tx, fi = s[WW].split(fi, nparts=blockdimx)
    s[WW].reorder(ty, tx, ci, fi)
    s[WW].bind(ty, thread_y)
    s[WW].bind(tx, thread_x)

    return s


def current_schedule(ops):
    output = ops[0]
    s = tvm.create_schedule(output)
    write_cache = s.cache_write(output.output(0), "local")
    A = output.input_tensors[0]
    W = output.input_tensors[1]
    AA = s.cache_read(A, "shared", [write_cache])
    AL = s.cache_read(AA, "local", [write_cache])
    WW = s.cache_read(W, "shared", [write_cache])
    WL = s.cache_read(WW, "local", [write_cache])

    rc = s[write_cache].op.reduce_axis[0]
    rco, rci = s[write_cache].split(rc, nparts=16)
    write_cache_rf = s.rfactor(write_cache, rco)
    # s[write_cache].bind(s[write_cache].op.reduce_axis[0], tvm.thread_axis("threadIdx.z"))
    s[write_cache_rf].compute_at(s[write_cache], s[write_cache].op.reduce_axis[0])

    bi, ci, hi, wi = s[output].op.axis
    b0, bi = s[output].split(bi, nparts=256)
    b1, bi = s[output].split(bi, nparts=1)
    b2, bi = s[output].split(bi, nparts=1)
    c0, ci = s[output].split(ci, nparts=8)
    c1, ci = s[output].split(ci, nparts=8)
    c2, ci = s[output].split(ci, nparts=8)
    # h0, hi = s[output].split(hi, nparts=1)
    # h1, hi = s[output].split(hi, nparts=14)
    # h2, hi = s[output].split(hi, nparts=1)
    # w0, wi = s[output].split(wi, nparts=1)
    # w1, wi = s[output].split(wi, nparts=1)
    # w2, wi = s[output].split(wi, nparts=14)
    hw = s[output].fuse(hi, wi)
    hi, wi = s[output].split(hw, factor=14)

    # s[output].reorder(b0, c0, h0, w0, b1, c1, h1, w1, b2, c2, h2, w2, bi, ci, hi, wi)
    # axis0 = s[output].fuse(b0, c0, h0, w0)
    # axis1 = s[output].fuse(b1, c1, h1, w1)
    # axis2 = s[output].fuse(b2, c2, h2, w2)
    # axis3 = s[output].fuse(bi, ci, hi, wi)

    s[output].bind(c0, tvm.thread_axis("blockIdx.x"))
    s[output].bind(b0, tvm.thread_axis("blockIdx.z"))
    s[output].bind(c1, tvm.thread_axis("vthread"))
    s[output].bind(hi, tvm.thread_axis("vthread"))
    s[output].bind(wi, tvm.thread_axis("threadIdx.x"))
    s[output].bind(c2, tvm.thread_axis("threadIdx.y"))

    s[write_cache].compute_at(s[output], wi)
    
    bi, ci, hi, wi = s[write_cache].op.axis
    # s[write_cache].reorder(rco, rcm, rci, bi, ci, hi, wi)

    for cache in [AA, WW]:
        s[cache].compute_at(s[write_cache_rf], s[write_cache_rf].op.axis[4])
    for cache in [AL, WL]:
       s[cache].compute_at(s[write_cache_rf], s[write_cache_rf].op.reduce_axis[0])
    # Schedule for A's shared memory load
    yi, xi, ci, ni = s[AA].op.axis
    cn = s[AA].fuse(ci, ni)
    tx, ci = s[AA].split(cn, nparts=14)
    ty, ni = s[AA].split(ci, nparts=8)
    # s[AA].reorder(ty, tx, yi, xi, ci, ni)
    s[AA].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[AA].bind(tx, tvm.thread_axis("threadIdx.x"))
    #s[AA].vectorize(ni)  # vectorize memory load

    # Schedule for W's shared memory load
    ci, fi= s[WW].op.axis
    ty, ci = s[WW].split(ci, nparts=8)
    tx, fi = s[WW].split(fi, nparts=14)
    # s[WW].reorder(ty, tx, ci, fi)
    s[WW].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[WW].bind(tx, tvm.thread_axis("threadIdx.x"))
    

    return s



def evaluate(s, bufs, target, dev_id, number=1):
    ctx = tvm.context(target, dev_id)
    tvm_arys = []
    for arg in bufs:
        shape = to_tuple(arg.shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(arg.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    stmt = tvm.lower(s, bufs, simple_mode=True)
    from flextensor.test.test_ir_visit_print import visit
    # visit(stmt, 0)
    func = tvm.build(s, bufs, target)
    print(func.imported_modules[0].get_source())
    evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
    time_cost = evaluator(*tvm_arys).mean * 1e3
    return time_cost


if __name__ == "__main__":
    shapes = (256, 256, 14, 14, 512)
    ops, bufs = conv2d_1x1(*shapes)
    s1 = jlc_schedule(ops)
    s2 = current_schedule(ops)
    cost1 = evaluate(s1, bufs, "cuda", 0, 10)
    print("s1: Time cost is %fms" % cost1)
    cost2 = evaluate(s2, bufs, "cuda", 0, 10)
    print("s2: Time cost is %fms" % cost2)