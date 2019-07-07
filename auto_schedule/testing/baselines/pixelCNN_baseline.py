import argparse
import timeit
import torch
import time
import tvm
import topi
import numpy as np
from auto_schedule.testing.configs.PixelCNN_config import PixelCNN_shape
from auto_schedule.testing.task import pixelcnn

torch.backends.cudnn.enabled = False


def pytorch_cpu(B, H, W, in_C, out_C, kernel_height, kernel_width, mask_type, bias, dilation, stride, padding, number=10, dev=0):
    Input = torch.rand([B, in_C, H, W], dtype=torch.float32)
    Kernel = torch.ones([out_C, in_C, kernel_height, kernel_width], dtype=torch.float32)
    assert mask_type in ['A', 'B']

    if mask_type == 'A':
        Kernel[:, :, kernel_height // 2, kernel_width // 2:] = 0
        Kernel[:, :, kernel_height // 2 + 1:, :] = 0
    elif mask_type == 'B':
        Kernel[:, :, kernel_height // 2, kernel_width // 2 + 1:] = 0
        Kernel[:, :, kernel_height // 2 + 1:, :] = 0

    conv = torch.nn.Conv2d(in_C, out_C, (kernel_height, kernel_width), stride=stride, dilation=dilation, padding=padding, bias=bias)
    if bias is not None:
        conv.bias = bias
    conv.weight.data = Kernel

    begin_time = time.time()
    for i in range(number):
        output = conv(Input)
    end_time = time.time()

    # ms
    return (end_time - begin_time) * 1e3 / number


def pytorch_cuda(B, H, W, C, out_C, kernel_height, kernel_width, mask_type, bias, dilation, stride, padding, number=10, dev=0):
    Input = torch.rand([B, C, H, W], dtype=torch.float32).cuda("cuda:" + str(dev))
    Kernel = torch.ones([out_C, C, kernel_height, kernel_width], dtype=torch.float32).cuda("cuda:" + str(dev))
    assert mask_type in ['A', 'B']

    if mask_type == 'A':
        Kernel[:, :, kernel_height // 2, kernel_width // 2:] = 0
        Kernel[:, :, kernel_height // 2 + 1:, :] = 0
    elif mask_type == 'B':
        Kernel[:, :, kernel_height // 2, kernel_width // 2 + 1:] = 0
        Kernel[:, :, kernel_height // 2 + 1:, :] = 0

    conv = torch.nn.Conv2d(C, out_C, (kernel_height, kernel_width), stride=stride, dilation=dilation, padding=padding, bias=bias)
    conv = conv.cuda("cuda:" + str(dev))
    if bias is not None:
        conv.bias = bias
    conv.weight.data = Kernel

    begin_time = time.time()
    for i in range(number):
        output = conv(Input)
    end_time = time.time()

    # ms
    return (end_time - begin_time) * 1e3 / number

def tvm_PixelCNN_cpu(B, H, W, C, out_C, kernel_height, kernel_width, mask_type, bias, dilation, stride, padding, number=10, dev=0):
    Input = torch.rand([B, H, W, C], dtype=torch.float32)
    Kernel = torch.zeros([out_C, C, kernel_height, kernel_width], dtype=torch.float32)

    s, bufs = pixelcnn(B, H, W, C, out_C, kernel_height, kernel_width, mask_type, bias, dilation=dilation, stride=stride, padding=padding)
    ctx = tvm.cpu(dev_id=dev)
    s =  tvm.create_schedule(s)
    f = tvm.build(s, bufs, "llvm")

    im = tvm.nd.array(Input.numpy().astype(np.float32), ctx)
    fi = tvm.nd.array(Kernel.numpy().astype(np.float32), ctx)

    in_height = H
    in_width = W
    out_height = (H + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (W + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    output_shape = (B, out_height, out_width, out_C)
    un = tvm.nd.array(np.zeros(output_shape).astype(np.float32), ctx)

    start_time = time.time()
    for i in range(number):
        f(im, fi, un)
    end_time = time.time()
    return (end_time - start_time) * 1e3 / number

if __name__ == "__main__":
    shapes = PixelCNN_shape

    """warm up"""
    """cost = pytorch_cpu(*shapes[0])
    cost = pytorch_cuda(*shapes[0])"""
    cost = tvm_PixelCNN_cpu(*shapes[0])
    # cost = tvm_PixelCNN_cuda(*shapes[0])

    for shape in shapes:
        print("Shape", shape)
        """cost = pytorch_cpu(*shape)
        print("Pytorch cost on cpu: {}ms".format(cost))
        cost = pytorch_cuda(*shape)
        print("Pytorch cost on cuda: {}ms".format(cost))"""
        cost = tvm_PixelCNN_cpu(*shape)
        print("Tvm cost on cpu: {}ms".format(cost))
        
    print("Done!")
