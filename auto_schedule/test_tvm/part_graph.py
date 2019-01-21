import tvm
from auto_schedule.ops import *


def main():
    A = tvm.placeholder((512, 512), dtype="float32", name="A")
    B = tvm.placeholder((512, 256), dtype="float32", name="B")
    C = matmul(A, B)
    W = tvm.placeholder((3, 3), dtype="float32", name="W")
    D = conv2d(C, W)
    E = conv2d(D, W)
    s = tvm.create_schedule(E.op)
    print(tvm.lower(s, [D, W, E], simple_mode=True))


if __name__ == "__main__":
    main()