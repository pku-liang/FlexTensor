import tvm
from tvm import autotvm
import topi
import sys
import time
import argparse
import os
import logging


def main(name, schedule_dispatcher, shapes, eval_func):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="Target Platform",
                        type=str, default='llvm')
    parser.add_argument("--dev", help="Device No.", type=int, default=0)
    # parser.add_argument(
    #     "-s", "--start", help="Shape index start", type=int, default=0)
    # parser.add_argument(
    #     "-e", "--end", help="Shape index end", type=int, default=-1)
    parser.add_argument("--trials", help="Trials", type=int, default=200)
    parser.add_argument("--timeout", help="Timeout", type=float, default=200)
    parser.add_argument("--verbose", help="Verbose", action="store_true")
    parser.add_argument("--tune", help="Tune", action="store_true")
    args = parser.parse_args()

    target = args.target
    dev = args.dev
    trials = args.trials
    timeout = args.timeout

    if args.verbose:
        logging.getLogger('autotvm').setLevel(logging.DEBUG)
        logging.getLogger('autotvm').addHandler(
            logging.StreamHandler(sys.stdout))

    schedule_func = schedule_dispatcher[target]

    for i, shape in enumerate(shapes):

        task = autotvm.task.create(
            schedule_func, args=shape, target=target)
        test_name = "{name}_{target}({dev})_case({no})_{shape}".format(
            name=name, target=target, dev=dev, no=i, shape=shape)
        print("test: {}".format(test_name))
        print("config_space: {}".format(task.config_space))

        logfile = "{}.log".format(test_name)

        if args.tune:
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(
                    repeat=3, min_repeat_ms=150, timeout=timeout)
            )
            if os.path.exists(logfile):
                os.remove(logfile)
            tuner = autotvm.tuner.XGBTuner(task)
            beg = time.time()
            tuner.tune(n_trial=trials, measure_option=measure_option,
                       callbacks=[autotvm.callback.log_to_file(logfile)], early_stopping=trials // 2)
            end = time.time()
            print("Optimize use {} s".format(end - beg))

        dispatch_context = autotvm.apply_history_best(logfile)
        best_config = dispatch_context.query(task.target, task.workload)    
        print("Best config: {}".format(best_config))

        with tvm.target.create(target):
            with autotvm.apply_history_best(logfile):
                s, bufs = schedule_func(*shape)
                func = tvm.build(s, bufs)

        cost = eval_func(func, shape, target, dev)
        print("Time Cost: {} ms".format(cost))
