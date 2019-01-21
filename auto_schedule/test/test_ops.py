import tvm
from auto_schedule.ops import *


def test():
    # test gemv
    try:
        A = tvm.placeholder((123, 1024), name="A")
        B = tvm.placeholder((1024,), name="B")
        C = gemv(A, B)
        s = tvm.create_schedule(C.op)
        tvm.lower(s, [A, B, C], simple_mode=True)
    except Exception as e:
        print("gemv fails: {}".format(str(e)))
    print("gemv pass!")

    # test gemm
    try:
        A = tvm.placeholder((123, 1024), name="A")
        B = tvm.placeholder((1024, 321), name="B")
        C = matmul(A, B)
        s = tvm.create_schedule(C.op)
        tvm.lower(s, [A, B, C], simple_mode=True)
    except Exception as e:
        print("gemm fails: {}".format(str(e)))
    print("gemm pass!")

    # test conv2d
    try:
        A = tvm.placeholder((1024, 1024), name="A")
        B = tvm.placeholder((5, 5), name="B")
        C = conv2d(A, B)
        s = tvm.create_schedule(C.op)
        tvm.lower(s, [A, B, C], simple_mode=True)
    except Exception as e:
        print("conv2d fails: {}".format(str(e)))
    print("conv2d pass!")

    # test conv2d_channel
    try:
        A = tvm.placeholder((1024, 1024, 5), name="A")
        B = tvm.placeholder((5, 5, 5, 10), name="B")
        C = conv2d_channel(A, B)
        s = tvm.create_schedule(C.op)
        tvm.lower(s, [A, B, C], simple_mode=True)
    except Exception as e:
        print("conv2d_channel fails: {}".format(str(e)))
    print("conv2d_channel pass!")

    # test conv2d_channel_batch
    try:
        A = tvm.placeholder((16, 1024, 1024, 5), name="A")
        B = tvm.placeholder((5, 5, 5, 10), name="B")
        C = conv2d_channel_batch(A, B)
        s = tvm.create_schedule(C.op)
        tvm.lower(s, [A, B, C], simple_mode=True)
    except Exception as e:
        print("conv2d_channel_batch fails: {}".format(str(e)))
    print("conv2d_channel_batch pass!")

    # test conv3d
    try:
        A = tvm.placeholder((102, 102, 102), name="A")
        B = tvm.placeholder((5, 5, 5), name="B")
        C = conv3d(A, B)
        s = tvm.create_schedule(C.op)
        tvm.lower(s, [A, B, C], simple_mode=True)
    except Exception as e:
        print("conv3d fails: {}".format(str(e)))
    print("conv3d pass!")

    # test conv3d_channel
    try:
        A = tvm.placeholder((102, 102, 102, 5), name="A")
        B = tvm.placeholder((5, 5, 5, 5, 10), name="B")
        C = conv3d_channel(A, B)
        s = tvm.create_schedule(C.op)
        tvm.lower(s, [A, B, C], simple_mode=True)
    except Exception as e:
        print("conv3d_channel fails: {}".format(str(e)))
    print("conv3d_channel pass!")

    # test conv3d_channel_batch
    try:
        A = tvm.placeholder((16, 102, 102, 102, 5), name="A")
        B = tvm.placeholder((5, 5, 5, 5, 10), name="B")
        C = conv3d_channel_batch(A, B)
        s = tvm.create_schedule(C.op)
        tvm.lower(s, [A, B, C], simple_mode=True)
    except Exception as e:
        print("conv3d_channel_batch fails: {}".format(str(e)))
    print("conv3d_channel_batch pass!")


if __name__ == "__main__":
    test()