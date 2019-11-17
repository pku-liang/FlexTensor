import os
import sys
import argparse
import time
import json
import tvm 
from auto_schedule.utils import Config
from auto_schedule.task import Task, TASK_TABLE
from auto_schedule.scheduler import schedule, schedule_with_config
from auto_schedule.measure import _evaluate
from auto_schedule.configs.grouped_config import grouped_shapes


def optimize(shapes, slevel=4, rlevel=3, target="llvm", dev_id=0, timeout=4.0, trials=100, parallel=1, 
        method="searching", use_model=False, logfile=sys.stdout):
    ret = dict()
    for i, shape in enumerate(shapes):
        print("Optimize grouped conv2d shape {}".format(shape), flush=True)
        batch, in_channel, H, W, out_channel, k, _, stride, padding, dilation, groups = shape
        # create an empty task but has the correct key we want
        task = Task(
            "conv2d",
            "grouped", 
            None, 
           (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, groups),
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
            parallel=parallel
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
        time_cost = _evaluate(s, bufs, target, task.dev_id, 10)
        print("Use", time_cost, "ms")
        print("Cost", end - beg, "s")
        print()
    return ret


def test(task_key, configs, dev_id=None):
    task = TASK_TABLE[task_key]
    s, bufs = schedule_with_config(task_key, configs)
    dev_id = dev_id if dev_id is not None else task.dev_id
    time_cost = _evaluate(s, bufs, task.target, dev_id, 10)
    print(task_key, "use", time_cost, "ms")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--from_", help="From which shape", type=int, default=0)
    parser.add_argument("-t", "--to", help="To which shape", type=int, default=-1)
    parser.add_argument("-l", "--log", help="Log file name", type=str, default="")
    parser.add_argument("--test", help="test file name", type=str, default="")
    parser.add_argument("--trials", help="number of trials for op", type=int, default=100)
    parser.add_argument("--target", help="target device type", type=str, default="llvm")
    parser.add_argument("--device", help="target device number", type=int, default=0)
    parser.add_argument("--timeout", help="timeout", type=float, default=4.0)
    parser.add_argument("--parallel", help="parallel", type=int, default=1)
    parser.add_argument("--use_model", help="use performance model", action="store_true")
    parser.add_argument("--method", help="how to schedule", type=str, default="searching")
    parser.add_argument("--slevel", type=int, default=4)
    parser.add_argument("--rlevel", type=int, default=3)
    args = parser.parse_args()
    shapes = grouped_shapes
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
                test(name, configs, args.device)

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
                logfile=flog
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
            logfile=sys.stdout
            )