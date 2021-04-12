import time
import tvm 
import torch
import numpy as np
from flextensor.configs.conv2d_config import all_conv_shapes
from flextensor.utils import to_tuple
from flextensor.nn import conv2d_nchw


def evaluate(s, bufs, target, dev_id, number=10):
    ctx = tvm.device(target, dev_id)
    tvm_arys = []
    for arg in bufs:
        shape = to_tuple(arg.shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(arg.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    func, evaluator = None, None
    try:
        func = tvm.build(s, bufs, target)
        # evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
        # time_cost = evaluator(*tvm_arys).mean * 1e3
        beg = time.time()
        for i in range(number):
            func(*tvm_arys)
        end = time.time()
        time_cost = (end - beg) * 1e3 / number
        return time_cost
    except Exception as e:
        print(e)
        return float("inf")


def pytorch_conv(batch, channel, out_channel, height, width, k_h, k_w, stride, pad, target, number=10):
    A = torch.rand((batch, channel, height, width), dtype=torch.float32)
    W = torch.rand((out_channel, channel, k_h, k_w), dtype=torch.float32)
    if target == "cuda":
        A = A.cuda()
        W = W.cuda()
    # warm-up
    Out = torch.nn.functional.conv2d(A, W, stride=stride, padding=pad)
    beg = time.time()
    for i in range(number):
        Out = torch.nn.functional.conv2d(A, W, stride=stride, padding=pad)
    end = time.time()
    return (end - beg) * 1e3 / number


def tvm_conv(batch, channel, out_channel, height, width, k_h, k_w, stride, pad, target, devid=0, number=10):
    A = tvm.te.placeholder((batch, channel, height, width), dtype="float32")
    W = tvm.te.placeholder((out_channel, channel, k_h, k_w), dtype="float32")
    Output = conv2d_nchw(A, W, stride=stride, padding=pad)
    s = tvm.te.create_schedule(Output.op)
    bufs = [A, W, Output]
    return evaluate(s, bufs, target, devid, number)


def compare(write_file):
    for config in all_conv_shapes:
        batch, channel, h, w, out_channel, _, k_h, k_w, _, stride, pad, _, _ = config
        torch_time = pytorch_conv(batch, channel, out_channel, h, w, k_h, k_w, stride, pad, "llvm", 10)
        tvm_time = tvm_conv(batch, channel, out_channel, h, w, k_h, k_w, stride, pad, "llvm", 1, 10)
        shape = (batch, channel, h, w, out_channel, k_h, k_w, stride, pad)
        print(shape, ": torch:", torch_time, "   tvm", tvm_time, file=write_file, flush=True)


if __name__ == "__main__":
    with open("cmp_conv_cpu.txt", "w") as f:
        compare(f)