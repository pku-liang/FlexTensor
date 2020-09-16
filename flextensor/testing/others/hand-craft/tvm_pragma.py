import tvm 
import numpy as np
import os


M = 1024
N = 512
K = 256


def test1():
    A = tvm.te.placeholder([M, K], name="A")
    B = tvm.te.placeholder([K, N], name="B")
    k = tvm.te.reduce_axis((0, K), name="k")
    # A1 = tvm.te.compute([M, K], lambda i, j: A[i, j], "A1")
    # B1 = tvm.te.compute([K, N], lambda i, j: B[i, j], "B1")
    C = tvm.te.compute([M, N], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), "C")

    s = tvm.te.create_schedule(C.op)

    A1 = s.cache_read(A, "local", [C])
    B1 = s.cache_read(B, "local", [C])

    m, n = s[C].op.axis
    om, im = s[C].split(m, nparts=1)
    s[C].bind(om, tvm.te.thread_axis("blockIdx.x"))
    mo, mi = s[C].split(im, factor=32)
    no, ni = s[C].split(n, factor=16)
    k = s[C].op.reduce_axis[0]
    ko, ki = s[C].split(k, factor=8)

    s[C].reorder(mo, no, ko, mi, ki, ni)
    # s[C].bind(no, tvm.te.thread_axis("threadIdx.x"))

    s[A1].compute_at(s[C], ko)
    s[B1].compute_at(s[C], ko)

    # print(tvm.lower(s, [A, B, C]))

    f = tvm.build(s, [A, B, C], target="opencl")
    print(dir(f))
    print(f.get_source())
    print(f.imported_modules[0].get_source())


def test2():
    tgt_host="llvm"
    tgt="aocl_sw_emu"
    n = tvm.te.var("n")
    A = tvm.te.placeholder((n,), name='A')
    B = tvm.te.placeholder((n,), name='B')
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name="C")

    s = tvm.te.create_schedule(C.op)
    px, x = s[C].split(C.op.axis[0], nparts=1)

    s[C].bind(px, tvm.te.thread_axis("pipeline"))

    fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

    fadd.save("myadd.o")
    fadd.imported_modules[0].save("myadd.aocx")

    tvm.contrib.cc.create_shared("myadd.so", ["myadd.o"])


def run_aocl():
    tgt="aocl_sw_emu"

    fadd = tvm.runtime.module.load_module("myadd.so")
    fadd_dev = tvm.runtime.module.load_module("myadd.aocx")
    fadd.import_module(fadd_dev)

    ctx = tvm.context(tgt, 0)

    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype("float32"), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype("float32"), ctx)
    c = tvm.nd.array(np.zeros(n, dtype="float32"), ctx)

    fadd(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())


def test3():
    A = tvm.te.placeholder([M, K], name="A")
    B = tvm.te.placeholder([K, N], name="B")
    k = tvm.te.reduce_axis((0, K), name="k")
    C = tvm.te.compute([M, N], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), "C")

    s = tvm.te.create_schedule(C.op)

    m, n = s[C].op.axis
    om, im = s[C].split(m, nparts=1)
    s[C].bind(om, tvm.te.thread_axis("pipeline"))
    mo, mi = s[C].split(im, factor=32)
    no, ni = s[C].split(n, factor=16)
    k = s[C].op.reduce_axis[0]
    ko, ki = s[C].split(k, factor=8)

    s[C].reorder(mo, no, ko, mi, ki, ni)

    print(tvm.lower(s, [A, B, C]))

    f = tvm.build(s, [A, B, C], target="sdaccel")
    print(dir(f))
    print(f.get_source())
    print(f.imported_modules[0].get_source())


def main():
    test1()
    # test3()


if __name__ == "__main__":
    main()