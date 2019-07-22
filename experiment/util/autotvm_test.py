import tvm
from tvm import autotvm
import topi
import sys
import time
import os
import logging


def run(name, schedule_func, schedule_args, eval_func, trials=100, timeout=200, target='llvm', dev=0, tune=True):
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)
    # logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    task = autotvm.task.create(
        schedule_func, args=schedule_args, target=target)
    print("config_space: {}".format(task.config_space))
    test_name = "{name}_{shape}".format(name=name, shape=schedule_args)
    logfile = "{}.log".format(test_name)

    if os.path.exists(logfile):
        os.remove(logfile)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            repeat=3, min_repeat_ms=150, timeout=timeout)
    )

    tuner = autotvm.tuner.XGBTuner(task)
    beg = time.time()
    if tune:
        tuner.tune(n_trial=trials, measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(logfile)], early_stopping=trials // 2)
    end = time.time()

    print("Optimize use {} s".format(end - beg))
    dispatch_context = autotvm.apply_history_best(logfile)
    best_config = dispatch_context.query(task.target, task.workload)
    print("Best config: {}".format(best_config))

    with autotvm.apply_history_best(logfile):
        with tvm.target.create(target):
            s, bufs = schedule_func(*schedule_args)
            func = tvm.build(s, bufs)

    eval_func(func, target, dev)
