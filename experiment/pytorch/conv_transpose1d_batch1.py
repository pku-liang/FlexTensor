from experiment.util.pytorch_test import test_pytorch
from experiment.util.autotvm_test import run
from experiment.shape import conv1d_batch1_shapes as shapes
import torch
import tvm
from tvm import autotvm


def pytorch_func(shape, target='llvm', dev=0):
    N, C, L, K, _, Lk, _, stride, padding, dilation, groups = shape
    L = (L + 2 * padding - dilation * (Lk - 1) - 1) // stride + 1
    C, K = K, C
    
    A, B = None, None

    def setup_gpu():
        nonlocal A, B
        A = torch.rand([N, C, L], dtype=torch.float32).cuda("cuda:" + str(dev))
        B = torch.rand([K, C//groups, Lk],
                       dtype=torch.float32).cuda("cuda:" + str(dev))

    def setup_cpu():
        nonlocal A, B
        A = torch.rand([N, C, L], dtype=torch.float32)
        B = torch.rand([K, C//groups, Lk], dtype=torch.float32)

    def stmt():
        nonlocal A, B
        torch.nn.functional.conv1d(
            A, B, stride=stride, padding=padding, dilation=dilation, groups=groups)

    if target == 'cuda':
        return setup_gpu, stmt
    else:
        return setup_cpu, stmt


if __name__ == "__main__":
    test_pytorch("conv_transpose1d_batch1", pytorch_func, shapes)
