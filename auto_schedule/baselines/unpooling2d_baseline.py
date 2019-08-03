import argparse
import timeit
import torch
import time
import tvm
import topi
import numpy as np
from auto_schedule.configs.maxunpooling2d_config import maxunpooling2d_shape
from auto_schedule.task import maxunpooling2d

torch.backends.cudnn.enabled = False


def pytorch_cpu(B, C, H, W, kernel_size, stride, padding, number=10, dev=0):
    Input = torch.rand([B, C, H, W], dtype=torch.float32)
    maxpool = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding, return_indices=True)
    Input, indices = maxpool(Input)

    begin_time = time.time()
    unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride, padding=padding)
    for i in range(number):
        output = unpool(Input, indices, output_size=(H, W))
    end_time = time.time()

    # ms
    return (end_time - begin_time) * 1e3 / number


def pytorch_cuda(B, C, H, W, kernel_size, stride, padding, number=10, dev=0):
    Input = torch.rand([B, C, H, W], dtype=torch.float32).cuda("cuda:" + str(dev))
    maxpool = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding, return_indices=True).cuda("cuda:" + str(dev))
    Input, indices = maxpool(Input)

    begin_time = time.time()
    unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride, padding=padding).cuda("cuda:" + str(dev))
    for i in range(number):
        output = unpool(Input, indices, output_size=(H, W))
    end_time = time.time()

    # ms
    return (end_time - begin_time) * 1e3 / number

def tvm_unpool2d_cpu(B, C, H, W, kernel_size, stride, padding, number=10, dev=0):
    Input = torch.rand([B, C, H, W], dtype=torch.float32).cuda("cuda:" + str(dev))
    maxpool = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding, return_indices=True).cuda("cuda:" + str(dev))
    Input, indices = maxpool(Input)
    Input = Input.cpu()
    indices = indices.cpu()

    s, bufs = maxunpooling2d(B, C, Input.shape[2], Input.shape[3], kernel_size, stride, padding)
    s =  tvm.create_schedule(s)
    print(tvm.lower(s.ops, bufs, simple_mode=True))
    ctx = tvm.cpu(dev)
    f = tvm.build(s, bufs, 'llvm')

    im = tvm.nd.array(Input.numpy().astype(np.float32), ctx)
    fi = tvm.nd.array(indices.numpy().astype(np.float32), ctx)

    in_height = Input.shape[2]
    in_width = Input.shape[3]
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size
    output_shape = (B, C, out_height, out_width)
    un = tvm.nd.array(np.zeros(output_shape).astype(np.float32), ctx)

    start_time = time.time()
    for i in range(number):
        f(im, fi, un)
    end_time = time.time()
    return (end_time - start_time) * 1e3 / number

def tvm_unpool2d_cuda(B, C, H, W, kernel_size, stride, padding, number=10, dev=0):
    Input = torch.rand([B, C, H, W], dtype=torch.float32).cuda("cuda:" + str(dev))
    maxpool = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding, return_indices=True).cuda("cuda:" + str(dev))
    Input, indices = maxpool(Input)
    Input = Input.cpu()
    indices = indices.cpu()

    s, bufs = maxunpooling2d(B, C, Input.shape[2], Input.shape[3], kernel_size, stride, padding)
    s =  tvm.create_schedule(s)
    f = tvm.build(s, bufs, "cuda")
    ctx = tvm.context("cuda", dev_id=dev)

    im = tvm.nd.array(Input.numpy().astype(np.float32), ctx)
    fi = tvm.nd.array(indices.numpy().astype(np.float32), ctx)

    in_height = Input.shape[2]
    in_width = Input.shape[3]
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size
    output_shape = (B, C, out_height, out_width)
    un = tvm.nd.array(np.zeros(output_shape).astype(np.float32), ctx)

    start_time = time.time()
    for i in range(number):
        f(im, fi, un)
    end_time = time.time()
    return (end_time - start_time) * 1e3 / number

if __name__ == "__main__":
    shapes = maxunpooling2d_shape

    """warm up"""
    cost = pytorch_cpu(*shapes[0])
    cost = pytorch_cuda(*shapes[0])
    # cost = tvm_unpool2d_cpu(*shapes[0])
    # cost = tvm_unpool2d_cuda(*shapes[0])

    for shape in shapes:
        print("Shape", shape)
        cost = pytorch_cpu(*shape)
        print("Pytorch cost on cpu: {}ms".format(cost))
        cost = pytorch_cuda(*shape)
        print("Pytorch cost on cuda: {}ms".format(cost))
        # cost = tvm_unpool2d_cpu(*shape)
        # print("Tvm cost on cpu: {}ms".format(cost))
        
    print("Done!")
