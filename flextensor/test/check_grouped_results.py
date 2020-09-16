import torch
import json
import argparse
import numpy as np 
import tvm
from flextensor.configs.grouped_config import grouped_shapes
from flextensor.task import TASK_TABLE
from flextensor.scheduler import schedule_with_config, Config
from flextensor.measure import _evaluate
from flextensor.utils import test_allclose
torch.backends.cudnn.enabled = False


def pytorch_cuda(N, H, W, C, kernel_size, K, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
    A = torch.rand([N, C, H, W], dtype=torch.float32).cuda("cuda:" + str(dev))
    W = torch.rand([K, C//groups, kernel_size, kernel_size], dtype=torch.float32).cuda("cuda:" + str(dev))

    # warm-up
    torch.nn.functional.conv2d(A, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    torch.cuda.synchronize()
    sum_time = 0.0
    
    for i in range(number):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ans = torch.nn.functional.conv2d(A, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        sum_time += start.elapsed_time(end)
    return sum_time / number


def test(task_key, configs, dev_id=None, number=10):
    task = TASK_TABLE[task_key]
    s, bufs = schedule_with_config(task_key, configs)
    dev_id = dev_id if dev_id is not None else task.dev_id
    time_cost = _evaluate(s, bufs, task.target, dev_id, number)
    print(task_key, "use", time_cost, "ms")


def check_result(configs, shape, target="cuda", dev_id=0):
    ctx = tvm.context(target, dev_id)
    name, configs = configs
    batch, in_channel, H, W, out_channel, k, _, stride, padding, dilation, groups = shape
    A_np = np.random.uniform(-10, 10, size=[batch, in_channel, H, W]).astype("float32")
    A_tvm = tvm.nd.array(A_np, ctx)
    A_torch = torch.tensor(A_np)    # .cuda("cuda:" + str(dev_id))
    W_np = np.random.uniform(-10, 10, size=[out_channel, in_channel//groups, k, k]).astype("float32")
    W_tvm = tvm.nd.array(W_np, ctx)
    W_torch = torch.tensor(W_np)    # .cuda("cuda:" + str(dev_id))
    Output_torch = torch.nn.functional.conv2d(A_torch, W_torch, stride=stride, padding=padding, dilation=dilation, groups=groups)
    Output_np = np.zeros(Output_torch.shape).astype(np.float32)
    Output_tvm = tvm.nd.array(Output_np, ctx)
    s, bufs = schedule_with_config(name, configs)
    func = tvm.build(s, bufs, target)
    func(A_tvm, W_tvm, Output_tvm)
    passed = test_allclose(Output_tvm.asnumpy(), Output_torch.cpu().numpy(), rtol=1e-5, print_diff=True)
    if passed == 1:
        print("Passed!")
    else:
        print("Failed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="test file name", type=str, default="")
    parser.add_argument("--device", help="target device number", type=int, default=0)
    parser.add_argument("-n", "--number", help="number test run", type=int, default=10)
    args = parser.parse_args()
    shapes = grouped_shapes
    
    if args.test != "":
        configs_lst = []
        with open(args.test, "r") as fin:
            for line in fin:
                name, string = line.split(":", 1)
                obj = json.loads(string)
                configs = Config(obj[0], obj[1])
                configs_lst.append((name, configs))
                test(name, configs, dev_id=args.device, number=args.number)
        for i, shape in enumerate(shapes):
            batch, in_channel, H, W, out_channel, k, _, stride, padding, dilation, groups = shape
            cost = pytorch_cuda(
                batch, H, W, in_channel, k, out_channel, 
                stride=stride, padding=padding, dilation=dilation, groups=groups, number=args.number, dev=args.device)
            print(i, shape, cost)
        for configs, shape in zip(configs_lst, shapes):
            check_result(configs, shape, dev_id=args.device)
