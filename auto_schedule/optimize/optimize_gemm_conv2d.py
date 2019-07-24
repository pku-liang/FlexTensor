import os
import sys
import argparse
import time
import json
import tvm 
import torch

from auto_schedule.scheduler import Config
from auto_schedule.task import Task, TASK_TABLE
from auto_schedule.scheduler import schedule, schedule_with_config
from auto_schedule.measure import _evaluate
from auto_schedule.configs.conv2d_config import *


shape_dict = {
    "yolo": yolo_shapes,
    "google": google_shapes,
    "squeeze": squeeze_shapes,
    "res": res_shapes,
    "vgg-16": vgg_16_shapes,
    "vgg-19": vgg_19_shapes
}


def optimize(prefix, from_, shapes, target="llvm", dev_id=0, trials=100, timeout=4.0, parallel=1, method="searching", use_model=False, logfile=sys.stdout):
    ret = dict()
    for i, shape in enumerate(shapes):
        print("Optimize {} convolution layer {} shape {}".format(prefix, i + 1 + from_, shape), flush=True)
        batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, _, _ = shape
        # create an empty task but has the correct key we want
        task = Task(
            "gemm_conv2d",
            prefix + str(i + from_), 
            None, 
            (batch, in_channel, height, width, out_channel, k_h, stride, padding, 1, 1), 
            target, 
            dev_id
            )

        beg = time.time()
        s, bufs, configs = schedule(
            task.key, 
            op_trial=trials, 
            timeout=timeout, 
            op_stop=30, 
            parallel=parallel, 
            method=method,
            use_model=use_model,
            trials=[trials//10, trials//5, trials, trials//5]
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
        print("######################################")
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
        print("######################################")
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

def schedule_with_config_local():
    with open("/home/retina/skw/work/AutoScheduler/gemm_conv.log", 'r') as f:
        

"""
(1, 3, 448, 448, 64, 3, 7, 7, 1, 2, 3, 1, 1)
    Conv 2d on cpu: 0.011640000343322753s
    Conv 2d on cuda: 0.006447720527648926s
(1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1)
    Conv 2d on cpu: 0.010520696640014648s
    Conv 2d on cuda: 0.006895184516906738s
(1, 192, 56, 56, 128, 192, 1, 1, 1, 1, 0, 1, 1)
    Conv 2d on cpu: 0.00572810173034668s
    Conv 2d on cuda: 0.005124855041503906s
(1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1)
    Conv 2d on cpu: 0.005372405052185059s
    Conv 2d on cuda: 0.003541111946105957s
(1, 256, 56, 56, 256, 256, 1, 1, 1, 1, 0, 1, 1)
    Conv 2d on cpu: 0.00752255916595459s
    Conv 2d on cuda: 0.0071736335754394535s
(1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1)
    Conv 2d on cpu: 0.014762544631958007s
    Conv 2d on cuda: 0.006854510307312012s
(1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1)
    Conv 2d on cpu: 0.0043433189392089845s
    Conv 2d on cuda: 0.0035385370254516603s
(1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1)
    Conv 2d on cpu: 0.005109810829162597s
    Conv 2d on cuda: 0.0018965244293212891s
(1, 512, 28, 28, 512, 512, 1, 1, 1, 1, 0, 1, 1)
    Conv 2d on cpu: 0.004613542556762695s
    Conv 2d on cuda: 0.003508114814758301s
(1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1)
    Conv 2d on cpu: 0.015011453628540039s
    Conv 2d on cuda: 0.0038038253784179687s
(1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1)
    Conv 2d on cpu: 0.003091883659362793s
    Conv 2d on cuda: 0.001864314079284668s
(1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1)
    Conv 2d on cpu: 0.007311129570007324s
    Conv 2d on cuda: 0.0012821674346923829s
(1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1)
    Conv 2d on cpu: 0.020050597190856934s
    Conv 2d on cuda: 0.0026390790939331056s
(1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1)
    Conv 2d on cpu: 0.0181943416595459s
    Conv 2d on cuda: 0.002562427520751953s
(1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1)
    Conv 2d on cpu: 0.018287014961242676s
    Conv 2d on cuda: 0.0017349958419799806s
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shapes", help="Use which shapes [yolo, google, res, squeeze, vgg-16, vgg-19]", type=str, default="")
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
    parser.add_argument("--test_torch", help="whether to test torch implementation", type=bool, default=False)
    args = parser.parse_args()
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
                    logfile=flog
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
                logfile=sys.stdout
                )
    if args.test != "":
        with open(args.test, "r") as fin:
            for line in fin:
                name, string = line.split(":", 1)
                obj = json.loads(string)
                configs = Config(obj[0], obj[1])
                test(name, configs, dev_id=args.device)
    
    if args.test_torch:
        assert args.shapes != ""
        shapes = shape_dict[args.shapes]
        """ Warm up """

        batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shapes[0]
        conv2d = torch.nn.Conv2d(in_channel, out_channel, (k_h, k_w), stride=stride, padding=padding, dilation=dilation, groups=groups).cuda()
        img = torch.rand((batch, in_channel, height, width)).cuda()
        res = conv2d(img)
        
        for shape in shapes:
            print(shape)
            batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
            start_time = time.time()
            conv2d = torch.nn.Conv2d(in_channel, out_channel, (k_h, k_w), stride=stride, padding=padding, dilation=dilation, groups=groups)
            for i in range(args.trials):
                img = torch.rand((batch, in_channel, height, width))
                res = conv2d(img)
            cpu_time = time.time() - start_time
            print("Conv 2d on cpu: {}s".format(cpu_time / args.trials))

            start_time = time.time()
            conv2d = conv2d.cuda()
            for i in range(args.trials):
                img = torch.rand((batch, in_channel, height, width)).cuda()
                res = conv2d(img)
            cuda_time = time.time() - start_time
            print("Conv 2d on cuda: {}s".format(cuda_time / args.trials))
