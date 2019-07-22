from experiment.util.pytorch_test import test_pytorch
from experiment.util.autotvm_test import run
from experiment.shape import group_conv2d_batch1_shapes as shapes
import torch
import tvm
from tvm import autotvm


def pytorch_func(shape, target='llvm', dev=0):
    N, C, H, W, K, Hk, _, stride, padding, dilation, groups = shape
    A, B = None, None

    def setup_gpu():
        nonlocal A, B
        A = torch.rand([N, C, H, W], dtype=torch.float32).cuda(
            "cuda:" + str(dev))
        B = torch.rand([K, C//groups, Hk, Hk],
                       dtype=torch.float32).cuda("cuda:" + str(dev))

    def setup_cpu():
        nonlocal A, B
        A = torch.rand([N, C, H, W], dtype=torch.float32)
        B = torch.rand([K, C//groups, Hk, Hk],
                       dtype=torch.float32)

    def stmt():
        nonlocal A, B
        torch.nn.functional.conv2d(
            A, B, stride=stride, padding=padding, dilation=dilation, groups=groups)

    if target == 'cuda':
        return setup_gpu, stmt
    else:
        return setup_cpu, stmt


if __name__ == "__main__":
    test_pytorch("group_conv2d_batch1", pytorch_func, shapes)
