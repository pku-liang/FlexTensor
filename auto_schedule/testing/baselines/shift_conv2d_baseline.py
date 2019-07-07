import argparse
import timeit
import torch
import time
import tvm
import topi
import numpy as np
import random
import math
from auto_schedule.testing.configs.shift_conv2d_config import shift_conv2d_shape
from auto_schedule.testing.task import shiftconv2d

torch.backends.cudnn.enabled = False

def tvm_shift_conv2d_cpu(B, H, W, C, kernel_size, dilation, stride=1, number=100, dev=0):
    Input = torch.rand([B, H, W, C], dtype=torch.float32)
    Kernel = torch.zeros([C, kernel_size, kernel_size], dtype=torch.float32)
    indexH = random.randint(0 ,kernel_size - 1)
    indexW = random.randint(0 ,kernel_size - 1)
    Kernel[:, indexH, indexW] = 0

    s, bufs = shiftconv2d(B, H, W, C, kernel_size, dilation, stride)
    s =  tvm.create_schedule(s)
    ctx = tvm.cpu(dev)
    # print(tvm.lower(s, bufs, simple_mode=True))
    f = tvm.build(s, bufs, 'llvm')

    im = tvm.nd.array(Input.numpy().astype(np.float32), ctx)
    fi = tvm.nd.array(Kernel.numpy().astype(np.float32), ctx)
    
    paddings = [math.ceil(((stride - 1) * H - stride + dilation * (kernel_size - 1)) / 2), 
                math.ceil(((stride - 1) * W - stride + dilation * (kernel_size - 1)) / 2)]

    image_height = H
    image_width = W
    out_height = math.floor((image_height + 2 * paddings[0]- dilation * (kernel_size - 1) - 1) / stride + 1)
    out_width = math.floor((image_width + 2 * paddings[1] - dilation * (kernel_size - 1) - 1) / stride + 1)
    output_shape = (B, out_height, out_width, C)
    un = tvm.nd.array(np.zeros(output_shape).astype(np.float32), ctx)

    start_time = time.time()
    for i in range(number):
        f(im, fi, un)
    end_time = time.time()
    return (end_time - start_time) * 1e3 / number

def tvm_shift_conv2d_cuda(B, H, W, C, kernel_size, stride, padding, number=100, dev=0):
    pass

if __name__ == "__main__":
    shapes = shift_conv2d_shape

    """warm up"""
    """cost = pytorch_cpu(*shapes[0])
    cost = pytorch_cuda(*shapes[0])
    cost = tvm_shift_conv2d_cpu(*shapes[0])"""
    # cost = tvm_shift_conv2d_cuda(*shapes[0])

    for shape in shapes:
        print("Shape", shape)
        cost = tvm_shift_conv2d_cpu(*shape)
        print("Tvm cost on cpu: {}ms".format(cost))
        
    print("Done!")
