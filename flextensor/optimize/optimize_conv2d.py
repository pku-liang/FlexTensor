import os
import sys
import argparse
import time
import json
import tvm
import numpy as np
from tvm import rpc
from tvm import te
from flextensor.utils import Config, RpcInfo
from flextensor.task import Task, TASK_TABLE
from flextensor.scheduler import schedule, schedule_with_config
from flextensor.utils import to_tuple, test_allclose
from flextensor.configs.conv2d_config import *
from flextensor.optimize.common import evaluate, test, init_arg_parser

shape_dict = {
    "yolo": yolo_shapes,
    "google": google_shapes,
    "squeeze": squeeze_shapes,
    "res": res_shapes,
    "vgg-16": vgg_16_shapes,
    "vgg-19": vgg_19_shapes,
    "yolo_b8": yolo_shapes_b8,
    "mobile_v2": mobilev2_shapes,
}


def optimize(prefix, from_, shapes, target="llvm", dev_id=0, trials=100, timeout=4.0,
             parallel=1, method="searching", use_model=False, rpc_info=None, force_inline=False, logfile=sys.stdout):
    ret = dict()
    for i, shape in enumerate(shapes):
        print("Optimize {} convolution layer {} shape {}".format(
            prefix, i + 1 + from_, shape), flush=True)
        batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
        # create an empty task but has the correct key we want
        task = Task(
            "conv2d",
            prefix + str(i + from_),
            None,
            (batch, in_channel, height, width, out_channel,
             k_h, stride, padding, dilation, groups),
            target,
            dev_id
        )
        beg = time.time()
        s, bufs, configs = schedule(
            task.key,
            op_trial=trials,
            op_stop=trials // 2,
            timeout=timeout,
            parallel=parallel,
            method=method,
            use_model=use_model,
            trials=[trials//10, trials],
            force_inline=force_inline,
            rpc_info=rpc_info,
            number=10,
        )
        end = time.time()
        # print(tvm.lower(s, bufs, simple_mode=True))
        print("######################################")
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
                             task.dev_id, 10, rpc_info)
        print("Use", time_cost, "ms")
        print("Cost", end - beg, "s")
        print()
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_arg_parser(parser)
    parser.add_argument(
        "-s", "--shapes", help="Use which shapes [yolo, google, res, squeeze, vgg-16, vgg-19]", type=str, default="")
    # parser.add_argument("--op_hint", type=str, default="split_fuse")
    args = parser.parse_args()
    rpc_info = RpcInfo(args.host, args.port, args.target_host,
                       args.device_key, args.use_rpc, args.fcompile, args.timeout)
    if args.shapes != "":
        shapes = shape_dict[args.shapes]
        if args.to < 0:
            end = len(shapes)
        else:
            end = args.to

        if args.log != "":
            with open(args.log, "a") as flog:
                ret = optimize(
                    args.shapes, args.from_,
                    shapes[args.from_:end],
                    target=args.target,
                    dev_id=args.device,
                    timeout=args.timeout,
                    trials=args.trials,
                    parallel=args.parallel,
                    method=args.method,
                    use_model=args.use_model,
                    rpc_info=rpc_info,
                    force_inline=args.force_inline,
                    logfile=flog,
                )
        else:
            ret = optimize(
                args.shapes,
                args.from_,
                shapes[args.from_:end],
                target=args.target,
                dev_id=args.device,
                timeout=args.timeout,
                trials=args.trials,
                parallel=args.parallel,
                method=args.method,
                use_model=args.use_model,
                rpc_info=rpc_info,
                force_inline=args.force_inline,
                logfile=sys.stdout,
            )
    if args.test != "":
        with open(args.test, "r") as fin:
            for line in fin:
                name, string = line.split(":", 1)
                obj = json.loads(string)
                configs = Config(obj[0], obj[1])
                test(name, configs, dev_id=args.device,
                     rpc_info=rpc_info, check_result=args.check)
