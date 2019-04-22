"""
Test operators in deep-learning.
Using PyTorch.

====================================
**Author**: `Size Zheng`
"""
import tvm
import torch
import numpy as np
from auto_schedule.testing.ops import *


def test_conv1d():
    #################################
    # test basic case
    inputs_np = np.random.random([4, 6, 14]).astype(np.float32) * 100
    weight_np = np.random.random([9, 2, 3]).astype(np.float32) * 100
    bias_np = np.random.random([9]).astype(np.float32) * 100
    
    inputs_torch = torch.tensor(inputs_np)
    weight_torch = torch.tensor(weight_np)
    bias_torch = torch.tensor(bias_np)
    output_torch = torch.nn.functional.conv1d(
        inputs_torch, weight_torch, bias_torch, stride=2, padding=1, dilation=2, groups=3)

    tvm_ctx = tvm.context("llvm", 0)
    inputs_tvm = tvm.nd.array(inputs_np, tvm_ctx)
    weight_tvm = tvm.nd.array(weight_np, tvm_ctx)
    bias_tvm = tvm.nd.array(bias_np, tvm_ctx)
    output_tvm = tvm.nd.array(np.zeros(output_torch.shape).astype(np.float32), tvm_ctx)
    inputs_t = tvm.placeholder(inputs_np.shape, dtype="float32")
    weight_t = tvm.placeholder(weight_np.shape, dtype="float32")
    bias_t = tvm.placeholder(bias_np.shape, dtype="float32")
    output_t = conv1d(inputs_t, weight_t, bias_t, stride=2, padding=1, dilation=2, groups=3)
    s = tvm.create_schedule(output_t.op)
    func = tvm.build(s, [inputs_t, weight_t, bias_t, output_t], "llvm")
    func(inputs_tvm, weight_tvm, bias_tvm, output_tvm)

    passed = test_allclose(output_tvm.asnumpy(), output_torch.numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Conv1d basic case passed!")
    else:
        print("Conv1d basic case failed!")


def test_conv_transpose1d():
    #################################
    # test basic case
    inputs_np = np.random.random([4, 9, 10]).astype(np.float32) * 100
    weight_np = np.random.random([9, 2, 3]).astype(np.float32) * 100
    bias_np = np.random.random([6]).astype(np.float32) * 100
    
    inputs_torch = torch.tensor(inputs_np)
    weight_torch = torch.tensor(weight_np)
    bias_torch = torch.tensor(bias_np)
    output_torch = torch.nn.functional.conv_transpose1d(
        inputs_torch, weight_torch, bias_torch, stride=2, padding=1, output_padding=1, dilation=1, groups=3)

    tvm_ctx = tvm.context("llvm", 0)
    inputs_tvm = tvm.nd.array(inputs_np, tvm_ctx)
    weight_tvm = tvm.nd.array(weight_np, tvm_ctx)
    bias_tvm = tvm.nd.array(bias_np, tvm_ctx)
    output_tvm = tvm.nd.array(np.zeros(output_torch.shape).astype(np.float32), tvm_ctx)
    inputs_t = tvm.placeholder(inputs_np.shape, dtype="float32")
    weight_t = tvm.placeholder(weight_np.shape, dtype="float32")
    bias_t = tvm.placeholder(bias_np.shape, dtype="float32")
    output_t = conv_transpose1d(inputs_t, weight_t, bias_t, stride=2, padding=1, output_padding=1, dilation=1, groups=3)
    s = tvm.create_schedule(output_t.op)
    func = tvm.build(s, [inputs_t, weight_t, bias_t, output_t], "llvm")
    func(inputs_tvm, weight_tvm, bias_tvm, output_tvm)

    passed = test_allclose(output_tvm.asnumpy(), output_torch.numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Conv_transpose1d basic case passed!")
    else:
        print("Conv_transpose1d basic case failed!")


def test_conv2d_nchw():
    #################################
    # test basic case
    inputs_np = np.random.random([4, 6, 7, 7]).astype(np.float32) * 100
    weight_np = np.random.random([9, 2, 3, 3]).astype(np.float32) * 100
    bias_np = np.random.random([9]).astype(np.float32) * 100
    
    inputs_torch = torch.tensor(inputs_np)
    weight_torch = torch.tensor(weight_np)
    bias_torch = torch.tensor(bias_np)
    output_torch = torch.nn.functional.conv2d(
        inputs_torch, weight_torch, bias_torch, stride=2, padding=1, dilation=2, groups=3)

    tvm_ctx = tvm.context("llvm", 0)
    inputs_tvm = tvm.nd.array(inputs_np, tvm_ctx)
    weight_tvm = tvm.nd.array(weight_np, tvm_ctx)
    bias_tvm = tvm.nd.array(bias_np, tvm_ctx)
    output_tvm = tvm.nd.array(np.zeros(output_torch.shape).astype(np.float32), tvm_ctx)
    inputs_t = tvm.placeholder(inputs_np.shape, dtype="float32")
    weight_t = tvm.placeholder(weight_np.shape, dtype="float32")
    bias_t = tvm.placeholder(bias_np.shape, dtype="float32")
    output_t = conv2d_nchw(inputs_t, weight_t, bias_t, stride=2, padding=1, dilation=2, groups=3)
    s = tvm.create_schedule(output_t.op)
    func = tvm.build(s, [inputs_t, weight_t, bias_t, output_t], "llvm")
    func(inputs_tvm, weight_tvm, bias_tvm, output_tvm)

    passed = test_allclose(output_tvm.asnumpy(), output_torch.numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Conv2d_nchw basic case passed!")
    else:
        print("Conv2d_nchw basic case failed!")


def test_conv_transpose2d_nchw():
    #################################
    # test basic case
    inputs_np = np.random.random([4, 9, 7, 7]).astype(np.float32) * 100
    weight_np = np.random.random([9, 2, 3, 3]).astype(np.float32) * 100
    bias_np = np.random.random([6]).astype(np.float32) * 100
    
    inputs_torch = torch.tensor(inputs_np)
    weight_torch = torch.tensor(weight_np)
    bias_torch = torch.tensor(bias_np)
    output_torch = torch.nn.functional.conv_transpose2d(
        inputs_torch, weight_torch, bias_torch, stride=2, padding=1, output_padding=1, dilation=2, groups=3)

    tvm_ctx = tvm.context("llvm", 0)
    inputs_tvm = tvm.nd.array(inputs_np, tvm_ctx)
    weight_tvm = tvm.nd.array(weight_np, tvm_ctx)
    bias_tvm = tvm.nd.array(bias_np, tvm_ctx)
    output_tvm = tvm.nd.array(np.zeros(output_torch.shape).astype(np.float32), tvm_ctx)
    inputs_t = tvm.placeholder(inputs_np.shape, dtype="float32")
    weight_t = tvm.placeholder(weight_np.shape, dtype="float32")
    bias_t = tvm.placeholder(bias_np.shape, dtype="float32")
    output_t = conv_transpose2d_nchw(inputs_t, weight_t, bias_t, stride=2, padding=1, 
                                    output_padding=1, dilation=2, groups=3)
    s = tvm.create_schedule(output_t.op)
    func = tvm.build(s, [inputs_t, weight_t, bias_t, output_t], "llvm")
    func(inputs_tvm, weight_tvm, bias_tvm, output_tvm)

    passed = test_allclose(output_tvm.asnumpy(), output_torch.numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Conv_transpose2d_nchw basic case passed!")
    else:
        print("Conv_transpose2d_nchw basic case failed!")


def test_depthwise_conv2d_nchw():
    #################################
    # test basic case
    inputs_np = np.random.random([4, 6, 7, 7]).astype(np.float32) * 100
    weight_np = np.random.random([18, 1, 3, 3]).astype(np.float32) * 100
    bias_np = np.random.random([18]).astype(np.float32) * 100
    
    inputs_torch = torch.tensor(inputs_np)
    weight_torch = torch.tensor(weight_np)
    bias_torch = torch.tensor(bias_np)
    output_torch = torch.nn.functional.conv2d(
        inputs_torch, weight_torch, bias_torch, stride=2, padding=1, dilation=2, groups=6)

    tvm_ctx = tvm.context("llvm", 0)
    inputs_tvm = tvm.nd.array(inputs_np, tvm_ctx)
    weight_tvm = tvm.nd.array(weight_np, tvm_ctx)
    bias_tvm = tvm.nd.array(bias_np, tvm_ctx)
    output_tvm = tvm.nd.array(np.zeros(output_torch.shape).astype(np.float32), tvm_ctx)
    inputs_t = tvm.placeholder(inputs_np.shape, dtype="float32")
    weight_t = tvm.placeholder(weight_np.shape, dtype="float32")
    bias_t = tvm.placeholder(bias_np.shape, dtype="float32")
    output_t = depthwise_conv2d_nchw(inputs_t, weight_t, bias_t, stride=2, padding=1, dilation=2)
    s = tvm.create_schedule(output_t.op)
    func = tvm.build(s, [inputs_t, weight_t, bias_t, output_t], "llvm")
    func(inputs_tvm, weight_tvm, bias_tvm, output_tvm)

    passed = test_allclose(output_tvm.asnumpy(), output_torch.numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Depthwise_conv2d_nchw basic case passed!")
    else:
        print("Depthwise_conv2d_nchw basic case failed!")


def test_gemm_conv2d_nchw():
    #################################
    # test basic case
    inputs_np = np.random.random([4, 6, 7, 7]).astype(np.float32) * 100
    weight_np = np.random.random([9, 2, 3, 3]).astype(np.float32) * 100
    bias_np = np.random.random([9]).astype(np.float32) * 100
    
    inputs_torch = torch.tensor(inputs_np)
    weight_torch = torch.tensor(weight_np)
    bias_torch = torch.tensor(bias_np)
    output_torch = torch.nn.functional.conv2d(
        inputs_torch, weight_torch, bias_torch, stride=2, padding=1, dilation=2, groups=3)

    tvm_ctx = tvm.context("llvm", 0)
    inputs_tvm = tvm.nd.array(inputs_np, tvm_ctx)
    weight_tvm = tvm.nd.array(weight_np, tvm_ctx)
    bias_tvm = tvm.nd.array(bias_np, tvm_ctx)
    output_tvm = tvm.nd.array(np.zeros(output_torch.shape).astype(np.float32), tvm_ctx)
    inputs_t = tvm.placeholder(inputs_np.shape, dtype="float32")
    weight_t = tvm.placeholder(weight_np.shape, dtype="float32")
    bias_t = tvm.placeholder(bias_np.shape, dtype="float32")
    output_t = gemm_conv2d_nchw(inputs_t, weight_t, bias_t, stride=2, padding=1, dilation=2, groups=3)
    s = tvm.create_schedule(output_t.op)
    func = tvm.build(s, [inputs_t, weight_t, bias_t, output_t], "llvm")
    func(inputs_tvm, weight_tvm, bias_tvm, output_tvm)

    passed = test_allclose(output_tvm.asnumpy(), output_torch.numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Gemm_conv2d_nchw basic case passed!")
    else:
        print("Gemm_conv2d_nchw basic case failed!")


def test_linear():
    #################################
    # test basic case
    inputs_np = np.random.random([2, 3, 2, 3, 17]).astype(np.float32) * 100
    weight_np = np.random.random([5, 17]).astype(np.float32) * 100
    bias_np = np.random.random([5]).astype(np.float32) * 100
    
    inputs_torch = torch.tensor(inputs_np)
    weight_torch = torch.tensor(weight_np)
    bias_torch = torch.tensor(bias_np)
    output_torch = torch.nn.functional.linear(inputs_torch, weight_torch, bias_torch)

    tvm_ctx = tvm.context("llvm", 0)
    inputs_tvm = tvm.nd.array(inputs_np, tvm_ctx)
    weight_tvm = tvm.nd.array(weight_np, tvm_ctx)
    bias_tvm = tvm.nd.array(bias_np, tvm_ctx)
    output_tvm = tvm.nd.array(np.zeros(output_torch.shape).astype(np.float32), tvm_ctx)
    inputs_t = tvm.placeholder(inputs_np.shape, dtype="float32")
    weight_t = tvm.placeholder(weight_np.shape, dtype="float32")
    bias_t = tvm.placeholder(bias_np.shape, dtype="float32")
    output_t = linear(inputs_t, weight_t, bias_t)
    s = tvm.create_schedule(output_t.op)
    func = tvm.build(s, [inputs_t, weight_t, bias_t, output_t], "llvm")
    func(inputs_tvm, weight_tvm, bias_tvm, output_tvm)

    passed = test_allclose(output_tvm.asnumpy(), output_torch.numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Linear basic case passed!")
    else:
        print("Linear basic case failed!")


def test_bilinear():
    #################################
    # test basic case
    inputs_np = np.random.random([2, 3, 2, 3, 17]).astype(np.float32) * 100
    another_np = np.random.random([2, 3, 2, 3, 8]).astype(np.float32) * 100
    weight_np = np.random.random([5, 17, 8]).astype(np.float32) * 100
    bias_np = np.random.random([5]).astype(np.float32) * 100
    
    inputs_torch = torch.tensor(inputs_np)
    another_torch = torch.tensor(another_np)
    weight_torch = torch.tensor(weight_np)
    bias_torch = torch.tensor(bias_np)
    output_torch = torch.nn.functional.bilinear(inputs_torch, another_torch, weight_torch, bias_torch)

    tvm_ctx = tvm.context("llvm", 0)
    inputs_tvm = tvm.nd.array(inputs_np, tvm_ctx)
    another_tvm = tvm.nd.array(another_np, tvm_ctx)
    weight_tvm = tvm.nd.array(weight_np, tvm_ctx)
    bias_tvm = tvm.nd.array(bias_np, tvm_ctx)
    output_tvm = tvm.nd.array(np.zeros(output_torch.shape).astype(np.float32), tvm_ctx)
    inputs_t = tvm.placeholder(inputs_np.shape, dtype="float32")
    another_t = tvm.placeholder(another_np.shape, dtype="float32")
    weight_t = tvm.placeholder(weight_np.shape, dtype="float32")
    bias_t = tvm.placeholder(bias_np.shape, dtype="float32")
    output_t = bilinear(inputs_t, another_t, weight_t, bias_t)
    s = tvm.create_schedule(output_t.op)
    func = tvm.build(s, [inputs_t, another_t, weight_t, bias_t, output_t], "llvm")
    func(inputs_tvm, another_tvm, weight_tvm, bias_tvm, output_tvm)

    passed = test_allclose(output_tvm.asnumpy(), output_torch.numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Bilinear basic case passed!")
    else:
        print("Bilinear basic case failed!")


def test_mean():
    #################################
    # test basic case
    inputs_np = np.random.random([2, 3, 27, 3, 17]).astype(np.float32) * 100
    
    inputs_torch = torch.tensor(inputs_np)
    output_torch = torch.mean(inputs_torch, dim=2)

    tvm_ctx = tvm.context("llvm", 0)
    inputs_tvm = tvm.nd.array(inputs_np, tvm_ctx)
    output_tvm = tvm.nd.array(np.zeros(output_torch.shape).astype(np.float32), tvm_ctx)
    inputs_t = tvm.placeholder(inputs_np.shape, dtype="float32")
    output_t = mean(inputs_t, dim=2)
    s = tvm.create_schedule(output_t.op)
    func = tvm.build(s, [inputs_t, output_t], "llvm")
    func(inputs_tvm, output_tvm)

    passed = test_allclose(output_tvm.asnumpy(), output_torch.numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Mean basic case passed!")
    else:
        print("Mean basic case failed!")


def test_variance():
    #################################
    # test basic case
    inputs_np = np.random.random([2, 3, 27, 3, 17]).astype(np.float32) * 100
    
    inputs_torch = torch.tensor(inputs_np)
    output_torch = inputs_torch.var(dim=2)

    tvm_ctx = tvm.context("llvm", 0)
    inputs_tvm = tvm.nd.array(inputs_np, tvm_ctx)
    output_tvm = tvm.nd.array(np.zeros(output_torch.shape).astype(np.float32), tvm_ctx)
    inputs_t = tvm.placeholder(inputs_np.shape, dtype="float32")
    output_t = variance(inputs_t, dim=2)
    s = tvm.create_schedule(output_t.op)
    func = tvm.build(s, [inputs_t, output_t], "llvm")
    func(inputs_tvm, output_tvm)

    passed = test_allclose(output_tvm.asnumpy(), output_torch.numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Variance basic case passed!")
    else:
        print("Variance basic case failed!")


def test():
    test_conv1d()
    test_conv_transpose1d()
    test_conv2d_nchw()
    test_conv_transpose2d_nchw()
    test_gemm_conv2d_nchw()
    test_linear()
    test_bilinear()
    test_mean()
    test_variance()


if __name__ == "__main__":
    print("Test begins...")
    test()
    print("Done.")