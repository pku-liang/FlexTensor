import os
import sys
import argparse
import time
import json
import tvm
import numpy as np
from tvm import rpc
from flextensor.utils import Config, RpcInfo
from flextensor.task import Task, TASK_TABLE
from flextensor.scheduler import schedule, schedule_with_config
from flextensor.measure import _evaluate
from flextensor.utils import to_tuple, test_allclose
from flextensor.configs.gemm_config import gemm_shapes
from flextensor.optimize.common import test, evaluate, init_arg_parser


def optimize(shapes, slevel=4, rlevel=3, target="llvm", dev_id=0, timeout=4.0, trials=100, parallel=1,
             method="searching", use_model=False, rpc_info=None, logfile=sys.stdout):
    ret = dict()
    for i, shape in enumerate(shapes):
        print("Optimize gemm shape %s [%.6f]" %
              (str(shape), time.time()), flush=True)
        N, K, M = shape
        # create an empty task but has the correct key we want
        task = Task(
            "gemm",
            "gemm",
            None,
            (N, K, M),
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
            op_stop=trials // 2,
            method=method,
            use_model=use_model,
            parallel=parallel,
            rpc_info=rpc_info,
            number=10
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
        time_cost = evaluate(task.key, s, bufs, target,
                             dev_id, rpc_info=rpc_info)
        print("Use", time_cost, "ms")
        print("Cost", end - beg, "s")
        print()
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_arg_parser(parser)
    args = parser.parse_args()
    shapes = [(2 ** n, 2 ** n, 2 ** n) for n in range(7, 13)] + [(256, 128, 128),
                                                                 (128, 256, 128), (128, 128, 256), (128, 256, 256), (256, 128, 256), (256, 256, 128)]
    rpc_info = RpcInfo(args.host, args.port, args.target_host,
                       args.device_key, args.use_rpc, args.fcompile, args.timeout)

    if args.to < 0:
        end = len(shapes)
    else:
        end = args.to

    if args.test != "":
        with open(args.test, "r") as fin:
            for line in fin:
                name, string = line.split(":", 1)
                obj = json.loads(string)
                configs = Config(obj[0], obj[1])
                test(name, configs, args.device, rpc_info=rpc_info, check_result=args.check)

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
                rpc_info=rpc_info
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
            rpc_info=rpc_info
        )
