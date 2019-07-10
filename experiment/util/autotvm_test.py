import tvm
from tvm import autotvm
import topi
import time


def run(name, schedule_func, schedule_args, eval_func, trials=100, timeout=4, number=10, target='llvm', dev=0, tune=True):
    task = autotvm.task.create(
        schedule_func, args=schedule_args, target=target)
    print("config_space length: {}".format(len(task.config_space)))
    test_name = "{name}_{shape}".format(name=name, shape=schedule_args)
    logfile = "{}.log".format(test_name)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=number, repeat=1, min_repeat_ms=150, timeout=timeout)
    )

    tuner = autotvm.tuner.XGBTuner(task)
    beg = time.time()
    if tune:
        tuner.tune(n_trials=trials, measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(logfile)])
    end = time.time()

    dispatch_context = autotvm.apply_history_best(logfile)
    best_config = dispatch_context.query(task.target, task.workload)
    print("Optimize use {} s".format(beg - end))
    print("Best config: {}".format(best_config))

    with autotvm.apply_history_best(logfile):
        with tvm.target.create(target):
            s, bufs = schedule_func(*schedule_args)
            func = tvm.build(s, bufs)

    return eval_func(func)
