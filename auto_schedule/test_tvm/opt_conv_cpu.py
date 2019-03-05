import numpy as np
import tvm
import multiprocessing as multi


# The sizes of inputs and filters
batch = 128
in_channel = 128
out_channel = 128
in_size = 14
kernel = 3
pad = 1
stride = 1
out_size = (in_size - kernel + 2*pad) // stride + 1


def conv_time(args):
    tile_x, tile_y, step, target, dev_id, number = args
    # Algorithm
    A = tvm.placeholder((in_size, in_size, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, out_channel), name='W')
    # Pad input
    Apad = tvm.compute(
        (in_size + 2*pad, in_size + 2*pad, in_channel, batch),
        lambda yy, xx, cc, nn: tvm.select(
            tvm.all(yy >= pad, yy - pad < in_size,
                    xx >= pad, xx - pad < in_size),
            A[yy - pad, xx - pad, cc, nn], tvm.const(0., "float32")),
        name='Apad')
    # Create reduction variables
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel), name='ry')
    rx = tvm.reduce_axis((0, kernel), name='rx')
    # Compute the convolution
    B = tvm.compute(
        (out_size, out_size, out_channel, batch),
        lambda yy, xx, ff, nn: tvm.sum(
            Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
            axis=[ry, rx, rc]),
        name='B')

    # Designate the memory hierarchy
    s = tvm.create_schedule(B.op)
    # s[Apad].compute_inline()
    BL = s.cache_write(B, "local")

    # tile consts
    # tile = 16
    # step = 16

    # Split the workloads
    hi, wi, fi, ni = s[B].op.axis
    bz = s[B].fuse(hi, wi)
    by, fi = s[B].split(fi, factor=tile_x)
    bx, ni = s[B].split(ni, factor=tile_y)
    s[B].reorder(bx, bz, by, fi, ni)
    bp = s[B].fuse(bz, bx)
    s[B].parallel(bp)
    # xi, yi, ci = s[B].op.reduce_axis
    # co, ci = s[B].split(ci, factor=step)
    # s[B].reorder(co, xi, yi, fi, ci, ni)
    # s[B].unroll(ci)
    # s[B].vectorize(ni)

    s[BL].compute_at(s[B], bp)
    h, w, f, n = s[BL].op.axis
    xi, yi, ci = s[BL].op.reduce_axis
    co, ci = s[BL].split(ci, factor=step)
    s[BL].reorder(co, xi, yi, f, ci, n)
    s[BL].unroll(ci)
    s[BL].vectorize(n)
    time_cost = evaluate(s, [A, W, B], target, dev_id, number)
    print("args={}, time_cost={}".format(args, time_cost))
    # stmt = tvm.lower(s, [A, W, B], simple_mode=True)
    # print(stmt)
    return time_cost


def evaluate(s, bufs, target, dev_id, number):
    A, W, B = bufs
    func = tvm.build(s, bufs, target)
    # print(func.imported_modules[0].get_source())
    if target == "cuda":
        ctx = tvm.gpu(dev_id)
    elif target == "llvm":
        ctx = tvm.cpu(dev_id)
    else:
        raise ValueError("not support {}".format(target))
    a_np = np.random.uniform(size=(in_size, in_size, in_channel, batch)).astype(A.dtype)
    w_np = np.random.uniform(size=(kernel, kernel, in_channel, out_channel)).astype(W.dtype)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros((out_size, out_size, out_channel, batch), dtype=B.dtype), ctx)
    # func(a, w, b)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
    return evaluator(a, w, b).mean * 1e3


def main():
    options = [2**i for i in range(3, 8)]
    args = []
    for tile_x in options:
        for tile_y in options:
            for step in options:
                args.append((tile_x, tile_y, step, "llvm", np.random.randint(0, 8), 1))
    pool = multi.Pool(10)
    pool_result = pool.map(conv_time, args)
    # pool_result.wait(timeout=300)
    tuple_lst = []
    for i, line in enumerate(pool_result):
        tuple_lst.append((i, line))
    with open("result.txt", "w") as f:
        tuple_lst = sorted(tuple_lst, key=lambda x: x[1])
        for (i, cost) in tuple_lst:
            f.write("args={}, time={}\n".format(args[i], cost))
    pool.close()


if __name__ == "__main__":
    # main()
    cost = conv_time((128, 64, 8, "llvm", 1, 10))
    print("{}ms".format(cost))
