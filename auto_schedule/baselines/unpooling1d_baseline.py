import argparse
import timeit
import torch
import time
import tvm
import topi
import numpy as np
from auto_schedule.configs.maxunpooling1d_config import maxunpooling1d_shape
from auto_schedule.task import maxunpooling1d

torch.backends.cudnn.enabled = False


def pytorch_cpu(B, C, L, kernel_size, stride, padding, number=10, dev=0):
    Input = torch.rand([B, C, L], dtype=torch.float32)
    maxpool = torch.nn.MaxPool1d(kernel_size, stride=stride, padding=padding, return_indices=True)
    Input, indices = maxpool(Input)

    begin_time = time.time()
    unpool = torch.nn.MaxUnpool1d(kernel_size, stride=stride, padding=padding)
    for i in range(number):
        output = unpool(Input, indices)
    end_time = time.time()

    # ms
    return (end_time - begin_time) * 1e3 / number


def pytorch_cuda(B, C, L, kernel_size, stride, padding, number=10, dev=0):
    Input = torch.rand([B, C, L], dtype=torch.float32).cuda("cuda:" + str(dev))
    maxpool = torch.nn.MaxPool1d(kernel_size, stride=stride, padding=padding, return_indices=True).cuda("cuda:" + str(dev))
    Input, indices = maxpool(Input)

    begin_time = time.time()
    unpool = torch.nn.MaxUnpool1d(kernel_size, stride=stride, padding=padding).cuda("cuda:" + str(dev))
    for i in range(number):
        output = unpool(Input, indices)
    end_time = time.time()

    # ms
    return (end_time - begin_time) * 1e3 / number

def tvm_unpool1d_cpu(B, C, L, kernel_size, stride, padding, number=10, dev=0):
    Input = torch.rand([B, C, L], dtype=torch.float32).cuda("cuda:" + str(dev))
    maxpool = torch.nn.MaxPool1d(kernel_size, stride=stride, padding=padding, return_indices=True).cuda("cuda:" + str(dev))
    Input, indices = maxpool(Input)
    Input = Input.cpu()
    indices = indices.cpu()

    s, bufs = maxunpooling1d(B, C, Input.shape[2],  kernel_size, stride, padding)
    s =  tvm.create_schedule(s)
    ctx = tvm.cpu(dev)
    f = tvm.build(s, bufs, 'llvm')

    im = tvm.nd.array(Input.numpy().astype(np.float32), ctx)
    fi = tvm.nd.array(indices.numpy().astype(np.float32), ctx)

    in_length = Input.shape[2]
    out_length = (in_length - 1) * stride - 2 * padding + kernel_size
    output_shape = (B, C, out_length)
    un = tvm.nd.array(np.zeros(output_shape).astype(np.float32), ctx)

    start_time = time.time()
    for i in range(number):
        f(im, fi, un)
    end_time = time.time()
    return (end_time - start_time) * 1e3 / number

def tvm_unpool1d_cuda(B, C, L, kernel_size, stride, padding, number=10, dev=0):
    Input = torch.rand([B, C, L], dtype=torch.float32).cuda("cuda:" + str(dev))
    maxpool = torch.nn.MaxPool1d(kernel_size, stride=stride, padding=padding, return_indices=True).cuda("cuda:" + str(dev))
    Input, indices = maxpool(Input)
    Input = Input.cpu()
    indices = indices.cpu()

    s, bufs = maxunpooling1d(B, C, Input.shape[2],  kernel_size, stride, padding)
    s =  tvm.create_schedule(s)
    f = tvm.build(s, bufs, "cuda")
    ctx = tvm.context("cuda", dev_id=dev)

    im = tvm.nd.array(Input.numpy().astype(np.float32), ctx)
    fi = tvm.nd.array(indices.numpy().astype(np.float32), ctx)

    in_length = Input.shape[2]
    out_length = (in_length - 1) * stride - 2 * padding + kernel_size
    output_shape = (B, C, out_length)
    un = tvm.nd.array(np.zeros(output_shape).astype(np.float32), ctx)

    start_time = time.time()
    for i in range(number):
        f(im, fi, un)
    end_time = time.time()
    return (end_time - start_time) * 1e3 / number

if __name__ == "__main__":
    shapes = maxunpooling1d_shape

    """warm up"""
    cost = pytorch_cpu(*shapes[0])
    cost = pytorch_cuda(*shapes[0])
    cost = tvm_unpool1d_cpu(*shapes[0])
    # cost = tvm_unpool1d_cuda(*shapes[0])

    for shape in shapes:
        print("Shape", shape)
        cost = pytorch_cpu(*shape)
        print("Pytorch cost on cpu: {}ms".format(cost))
        cost = pytorch_cuda(*shape)
        print("Pytorch cost on cuda: {}ms".format(cost))
        cost = tvm_unpool1d_cpu(*shape)
        print("Tvm cost on cpu: {}ms".format(cost))
        
    print("Done!")
