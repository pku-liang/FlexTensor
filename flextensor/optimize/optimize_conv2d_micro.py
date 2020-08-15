import os
import sys
import argparse
import time
import json
import tvm
import numpy as np
from tvm import micro, rpc
from flextensor.utils import Config, RpcInfo
from flextensor.task import Task, TASK_TABLE, register_task
from flextensor.scheduler import schedule, schedule_with_config
from flextensor.measure import _evaluate
from flextensor.utils import to_tuple
from flextensor.configs.conv2d_config import res_shapes_b1
from tvm.micro.base import compile_micro_mod
import json
from flextensor.utils import get_iter_info
from flextensor.ppa_model import measure_latency

LOCAL_RPC = False
LIB_DIR = "."


def conv2d_micro(N, C, H, W, K, Hk, Wk, S=1, P=0, D=1, dtype="int8"):
    H += 2 * P
    W += 2 * P
    Ho = (H - (Hk - 1) * D - 1) // S + 1
    Wo = (W - (Wk - 1) * D - 1) // S + 1
    Input = tvm.placeholder((N, H, W, C), name="Input", dtype=dtype)
    Filter = tvm.placeholder((Hk, Wk, C, K), name="Filter", dtype=dtype)
    rh = tvm.reduce_axis((0, Hk), name="rh")
    rw = tvm.reduce_axis((0, Wk), name="rw")
    rc = tvm.reduce_axis((0, C), name="rc")
    Output = tvm.compute(
        (N, Ho, Wo, K),
        lambda n, h, w, k: tvm.sum(
            Input[n, h + rh, w + rw, rc] * Filter[rh, rw, rc, k],
            axis=[rh, rw, rc]
        ), name="Output")

    return [Output.op], [Input, Filter, Output]


resnet_50_shapes = [
    (1, 3, 224, 224, 64, 7, 7, 2, 3, 1),
    (1, 64, 56, 56, 256, 1, 1, 1, 0, 1),
    (1, 64, 56, 56, 64, 1, 1, 1, 0, 1),
    (1, 64, 56, 56, 64, 3, 3, 1, 1, 1),
    (1, 64, 56, 56, 256, 1, 1, 1, 0, 1),
    (1, 256, 56, 56, 64, 1, 1, 1, 0, 1),
    (1, 64, 56, 56, 64, 3, 3, 1, 1, 1),
    (1, 64, 56, 56, 256, 1, 1, 1, 0, 1),
    (1, 256, 56, 56, 64, 1, 1, 1, 0, 1),
    (1, 64, 56, 56, 64, 3, 3, 1, 1, 1),
    (1, 64, 56, 56, 256, 1, 1, 1, 0, 1),
    (1, 256, 56, 56, 512, 1, 1, 2, 0, 1),
    (1, 256, 56, 56, 128, 1, 1, 2, 0, 1),
    (1, 128, 28, 28, 128, 3, 3, 1, 1, 1),
    (1, 128, 28, 28, 512, 1, 1, 1, 0, 1),
    (1, 512, 28, 28, 128, 1, 1, 1, 0, 1),
    (1, 128, 28, 28, 128, 3, 3, 1, 1, 1),
    (1, 128, 28, 28, 512, 1, 1, 1, 0, 1),
    (1, 512, 28, 28, 128, 1, 1, 1, 0, 1),
    (1, 128, 28, 28, 128, 3, 3, 1, 1, 1),
    (1, 128, 28, 28, 512, 1, 1, 1, 0, 1),
    (1, 512, 28, 28, 128, 1, 1, 1, 0, 1),
    (1, 128, 28, 28, 128, 3, 3, 1, 1, 1),
    (1, 128, 28, 28, 512, 1, 1, 1, 0, 1),
    (1, 512, 28, 28, 1024, 1, 1, 2, 0, 1),
    (1, 512, 28, 28, 256, 1, 1, 2, 0, 1),
    (1, 256, 14, 14, 256, 3, 3, 1, 1, 1),
    (1, 256, 14, 14, 1024, 1, 1, 1, 0, 1),
    (1, 1024, 14, 14, 256, 1, 1, 1, 0, 1),
    (1, 256, 14, 14, 256, 3, 3, 1, 1, 1),
    (1, 256, 14, 14, 1024, 1, 1, 1, 0, 1),
    (1, 1024, 14, 14, 256, 1, 1, 1, 0, 1),
    (1, 256, 14, 14, 256, 3, 3, 1, 1, 1),
    (1, 256, 14, 14, 1024, 1, 1, 1, 0, 1),
    (1, 1024, 14, 14, 256, 1, 1, 1, 0, 1),
    (1, 256, 14, 14, 256, 3, 3, 1, 1, 1),
    (1, 256, 14, 14, 1024, 1, 1, 1, 0, 1),
    (1, 1024, 14, 14, 256, 1, 1, 1, 0, 1),
    (1, 256, 14, 14, 256, 3, 3, 1, 1, 1),
    (1, 256, 14, 14, 1024, 1, 1, 1, 0, 1),
    (1, 1024, 14, 14, 256, 1, 1, 1, 0, 1),
    (1, 256, 14, 14, 256, 3, 3, 1, 1, 1),
    (1, 256, 14, 14, 1024, 1, 1, 1, 0, 1),
    (1, 1024, 14, 14, 2048, 1, 1, 2, 0, 1),
    (1, 1024, 14, 14, 512, 1, 1, 2, 0, 1),
    (1, 512, 7, 7, 512, 3, 3, 1, 1, 1),
    (1, 512, 7, 7, 2048, 1, 1, 1, 0, 1),
    (1, 2048, 7, 7, 512, 1, 1, 1, 0, 1),
    (1, 512, 7, 7, 512, 3, 3, 1, 1, 1),
    (1, 512, 7, 7, 2048, 1, 1, 1, 0, 1),
    (1, 2048, 7, 7, 512, 1, 1, 1, 0, 1),
    (1, 512, 7, 7, 512, 3, 3, 1, 1, 1),
    (1, 512, 7, 7, 2048, 1, 1, 1, 0, 1),
]

# resnet_50_shapes = [
#     (1, 64, 56, 56, 64, 3, 3, 1, 1, 1),
# ]

for s in resnet_50_shapes:
    for j in range(4):
        register_task(Task("conv2d", "conv2d_micro",
                           conv2d_micro, (*s, "int8"), "micro", j))

MAESTRO_PATH = "/home/herlight/IT/micro/maestro/"


def evaluate_micro(name, s, bufs, dev_id, number, rpc_info):
    assert rpc_info is not None

    info = get_iter_info(s)
    with open("./info.json", mode="w") as fp:
        json.dump(info, fp)
    return measure_latency(info)

    tracker = rpc.connect_tracker(rpc_info.server_ip, rpc_info.server_port)
    remote = tracker.request(rpc_info.device_key, priority=1,
                             session_timeout=10000)
    ctx = remote.context(rpc_info.target, dev_id)
    tvm_arys = []
    for buf in bufs:
        shape = to_tuple(buf.shape)
        tmp = np.random.uniform(-128, 128, size=shape).astype(buf.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    try:
        func_file = "{}.obj".format(name)
        func = tvm.build(s, bufs, target=rpc_info.target)
        compile_micro_mod(func_file,
                          func, rpc_info.micro_device_config,
                          aux_sources=rpc_info.aux_sources,
                          aux_options=rpc_info.aux_options)
        remote.upload(os.path.join(LIB_DIR, func_file))
        func = remote.load_module(func_file)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
        time_cost = evaluator(*tvm_arys).mean
    except Exception as e:
        print(e)
        time_cost = float("inf")
    finally:
        while len(tvm_arys) > 0:
            del tvm_arys[-1]
        if os.path.exists(os.path.join(LIB_DIR, func_file)):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
        elif os.path.exists(os.path.join(LIB_DIR, func_file + ".so")):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
    return time_cost


def evaluate(name, s, bufs, target, dev_id, number, rpc_info):
    if rpc_info is not None:
        host = rpc_info.host
        port = rpc_info.port
    else:
        # local
        host = "0.0.0.0"
        port = 9090  # default port
    if host == "0.0.0.0":
        if LOCAL_RPC:
            use_rpc = True
        else:
            use_rpc = False
    else:
        use_rpc = True
    if use_rpc:
        remote = rpc.connect(host, port)
        ctx = remote.context(target, dev_id)
    else:
        ctx = tvm.context(target, dev_id)
    tvm_arys = []
    for buf in bufs:
        shape = to_tuple(buf.shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(buf.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    try:
        func_file = "{}.tar".format(name)
        if rpc_info is not None and rpc_info.target_host is not None:
            func = tvm.build(s, bufs, target=target,
                             target_host=rpc_info.target_host)
        else:
            func = tvm.build(s, bufs, target=target)
        if use_rpc:
            func.export_library(os.path.join(LIB_DIR, func_file))
            remote.upload(os.path.join(LIB_DIR, func_file))
            func = remote.load_module(func_file)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
        time_cost = evaluator(*tvm_arys).mean
    except Exception as e:
        print(e)
        time_cost = float("inf")
    finally:
        while len(tvm_arys) > 0:
            del tvm_arys[-1]
        if os.path.exists(os.path.join(LIB_DIR, func_file)):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
        elif os.path.exists(os.path.join(LIB_DIR, func_file + ".so")):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
    return time_cost


def optimize(shapes, slevel=4, rlevel=3, target="llvm", dev_id=0, timeout=4.0, trials=100, parallel=1,
             method="searching", use_model=False, rpc_info=None, logfile=sys.stdout, dtype="float32"):
    ret = dict()
    fp = open("result.log", "w")
    for i, shape in enumerate(shapes):
        print("Optimize conv2d-micro shape %s [%.6f]" %
              (str(shape), time.time()), flush=True)
        # create an empty task but has the correct key we want
        task = Task(
            "conv2d",
            "conv2d_micro",
            None,
            (*shape, dtype),
            target,
            dev_id
        )
        beg = time.time()
        s, bufs, configs = schedule(
            task.key,
            slevel=slevel,
            rlevel=rlevel,
            op_trial=trials,
            timeout=timeout,
            op_stop=30,
            method=method,
            use_model=use_model,
            parallel=parallel,
            rpc_info=rpc_info,
            model_measurer=measure_latency
        )
        end = time.time()
        # print(tvm.lower(s, bufs, simple_mode=True))
        print("###################################### [%.6f]" % time.time())
        print("op schedules:")
        for config in configs.op_config_lst:
            print("----------------------------------")
            for name, value in config.items():
                if value:
                    print(name, value)
        print("graph schedules:")
        for name, value in configs.graph_config.items():
            if value:
                print(name, value)
        ret[task.key] = configs
        string = json.dumps(configs)
        line = task.key + ":" + string
        print(line, file=logfile, flush=True)
        s, bufs = schedule_with_config(task.key, configs)
        time_cost = evaluate_micro(task.key, s, bufs, dev_id, 1, rpc_info)
        print("Use", time_cost)
        print("Cost", end - beg, "s")
        fp.write(json.dumps({
            "shape": list(shape),
            "info": get_iter_info(s),
            "cost": float(time_cost)
        }) + "\n")
        print()
    fp.close()
    return ret


def test(task_key, configs, dev_id=None, rpc_info=None):
    task = TASK_TABLE[task_key]
    s, bufs = schedule_with_config(task_key, configs)
    print(tvm.lower(s, bufs, simple_mode=True))
    dev_id = dev_id if dev_id is not None else task.dev_id
    if task.target == "micro":
        time_cost = evaluate_micro(task_key, s, bufs, dev_id, 10, rpc_info)
    else:
        time_cost = evaluate(
            task_key, s, bufs, task.target, dev_id, 10, rpc_info)
    print(task_key, "use", time_cost)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--timeout", help="timeout", type=float, default=4.0)
    parser.add_argument("--parallel", help="parallel", type=int, default=1)
    parser.add_argument(
        "--use_model", help="use performance model", action="store_true")
    parser.add_argument("--micro", help="build for micro", default="")
    parser.add_argument("--method", help="how to schedule",
                        type=str, default="searching")
    parser.add_argument("--slevel", type=int, default=4)
    parser.add_argument("--rlevel", type=int, default=3)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--target_host", type=str, default="llvm")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()
    shapes = resnet_50_shapes
    rpc_info = RpcInfo(args.host, args.port, args.target_host)

    if args.micro == "spike":
        intrinsic_filename = "kernel.c"
        zynq_host = "127.0.0.1"
        rpc_info.target = "c -device=micro_dev"
        rpc_info.micro_device_config = micro.device.riscv.spike.default_config(
            0x800000000, zynq_host, 6666)
        rpc_info.aux_sources = [f"{os.path.realpath(intrinsic_filename)}"]
        rpc_info.aux_options = [
            f"-I{os.path.dirname(os.path.realpath(intrinsic_filename))}"]
        rpc_info.server_ip = "127.0.0.1"
        rpc_info.server_port = 9190
        rpc_info.device_key = "spike"
        rpc_info.host = zynq_host
    elif args.micro == "zync":
        intrinsic_filename = "kernel.c"
        zynq_host = "127.0.0.1"
        rpc_info.target = "c -device=micro_dev"
        rpc_info.micro_device_config = micro.device.riscv.rocketchip_zynq.default_config(
            0x800000000, zynq_host, 6666)
        rpc_info.aux_sources = [f"{os.path.realpath(intrinsic_filename)}"]
        rpc_info.aux_options = [
            f"-I{os.path.dirname(os.path.realpath(intrinsic_filename))}"]
        rpc_info.server_ip = "127.0.0.1"
        rpc_info.server_port = 9190
        rpc_info.device_key = "gemmini"
        rpc_info.host = zynq_host
    elif args.micro != "":
        raise ValueError("Not support micro %s" % args.micro)

    if args.to < 0:
        end = len(shapes)
    else:
        end = args.to

    if args.test != "":
        with open(args.test, "r") as fin:
            lines = [l for l in fin]
            for line in lines[args.from_:end]:
                name, string = line.split(":", 1)
                obj = json.loads(string)
                configs = Config(obj[0], obj[1])
                test(name, configs, args.device, rpc_info=rpc_info)

    elif args.log != "":
        with open(args.log, "a") as flog:
            ret = optimize(
                shapes[args.from_:end],
                slevel=args.slevel,
                rlevel=args.rlevel,
                target=args.target,
                dev_id=args.device,
                timeout=args.timeout,
                trials=args.trials,
                parallel=args.parallel,
                use_model=args.use_model,
                method=args.method,
                logfile=flog,
                rpc_info=rpc_info,
                dtype=args.dtype
            )
    else:
        ret = optimize(
            shapes[args.from_:end],
            slevel=args.slevel,
            rlevel=args.rlevel,
            target=args.target,
            dev_id=args.device,
            timeout=args.timeout,
            trials=args.trials,
            parallel=args.parallel,
            use_model=args.use_model,
            method=args.method,
            logfile=sys.stdout,
            rpc_info=rpc_info,
            dtype=args.dtype
        )
