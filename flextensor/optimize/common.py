import os
import sys
import argparse
import time
import json
import tvm
from tvm import te
import numpy as np
from flextensor.utils import Config, RpcInfo, to_tuple, test_allclose
from flextensor.task import Task, TASK_TABLE
from flextensor.scheduler import schedule, schedule_with_config
from flextensor.configs.conv1d_config import conv1d_shapes

LIB_DIR = "."


def evaluate(name, s, bufs, target, dev_id, number=10, rpc_info=None, result_generator=None):
    if rpc_info is not None:
        use_rpc = rpc_info.use_rpc
        target_host = rpc_info.target_host
        fcompile = rpc_info.fcompile
    else:
        use_rpc, target_host, fcompile = None, None, None

    remote = rpc_info.get_remote()
    dev = (remote if remote else tvm).device(target, dev_id)

    np_arys = [
        np.random.uniform(-10, 10, size=to_tuple(buf.shape)).astype(buf.dtype) for buf in bufs]
    tvm_arys = [tvm.nd.array(arr, dev) for arr in np_arys]
    func_file = f"{name}.so"
    time_cost = float("inf")
    try:
        func = tvm.build(s, bufs, target=target, target_host=target_host)
        if use_rpc:
            func.export_library(os.path.join(LIB_DIR, func_file), fcompile)
            remote.upload(os.path.join(LIB_DIR, func_file))
            func = remote.load_module(func_file)
        func(*tvm_arys)
        if result_generator is not None:
            print("Test whether computed...")
            result = tvm_arys[-1].asnumpy()
            test_allclose(result, np_arys[-1], rtol=1e-3, print_diff=True)
            print("Test correctness...")
            expected = result_generator(np_arys)
            test_allclose(result, expected, rtol=1e-3, print_diff=True)
        evaluator = func.time_evaluator(func.entry_name, dev, number=number)
        time_cost = evaluator(*tvm_arys).mean * 1e3
    except Exception as e:
        print(e)
    finally:
        while len(tvm_arys) > 0:
            del tvm_arys[-1]
        if os.path.exists(os.path.join(LIB_DIR, func_file)):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)

    return time_cost


def test(task_key, configs, dev_id=None, rpc_info=None, check_result=False):
    task = TASK_TABLE[task_key]
    result_generator = None
    if check_result:
        def _generator(np_arys):
            ops, bufs = task.func(*task.args)
            s = te.create_schedule(ops)
            func = tvm.build(s, bufs, target="llvm")
            ctx = tvm.cpu(0)
            tvm_arys = [tvm.nd.array(arr, ctx) for arr in np_arys]
            func(*tvm_arys)
            return tvm_arys[-1].asnumpy()
        result_generator = _generator

    s, bufs = schedule_with_config(task_key, configs)
    dev_id = dev_id if dev_id is not None else task.dev_id
    time_cost = evaluate(task_key, s, bufs, task.target,
                         dev_id, 100, rpc_info, result_generator)
    print(task_key, "use", time_cost, "ms")
    print()


def init_arg_parser(parser):
    parser.add_argument(
        "-f", "--from_", help="From which shape", type=int, default=0)
    parser.add_argument("-t", "--to", help="To which shape",
                        type=int, default=-1)
    parser.add_argument("-l", "--log", help="Log file name",
                        type=str, default="")
    parser.add_argument("--test", help="test file name", type=str, default="")
    parser.add_argument(
        "--trials", help="number of trials for op", type=int, default=100)
    parser.add_argument("--target", help="target device type",
                        type=str, default="llvm")
    parser.add_argument(
        "--device", help="target device number", type=int, default=0)
    parser.add_argument("--timeout", help="timeout", type=int, default=4)
    parser.add_argument("--parallel", help="parallel", type=int, default=1)
    parser.add_argument(
        "--use_model", help="use performance model", action="store_true")
    parser.add_argument("--method", help="how to schedule",
                        type=str, default="searching")
    parser.add_argument("--slevel", type=int, default=4)
    parser.add_argument("--rlevel", type=int, default=3)
    parser.add_argument("--force_inline", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--target_host", type=str, default="llvm")
    parser.add_argument("--device_key", type=str, default="")
    parser.add_argument("--use_rpc", type=str, choices=["server", "tracker"])
    parser.add_argument("--fcompile", type=str, choices=["ndk"])
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--check", action="store_true")
