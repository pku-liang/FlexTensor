import tvm
import os
import numpy as np
import argparse
from tvm import relay
from tvm.relay import testing
from tvm.relay.testing.init import create_workload
from tvm.relay.testing import layers
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime


def get_operator(data_shape, out_channel, kernel_size, strides, padding, dtype="float32"):
    data = relay.var("data", shape=data_shape, dtype=dtype)
    body = layers.conv2d(data=data, channels=out_channel, kernel_size=kernel_size, strides=strides, padding=padding, name="conv2d")
    return relay.Function(relay.ir_pass.free_vars(body), body)


def get_workload(batch_size, image_shape, out_channel, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), dtype="float32"):
    data_shape = (batch_size, *image_shape)
    op = get_operator(data_shape, out_channel, kernel_size, strides, padding, dtype=dtype)
    sym, params = create_workload(op)
    return sym, params, data_shape


target = tvm.target.cuda()
batch_size = 1
image_shape = (1024, 7, 7)
out_channel = 1024
kernel_size = (3, 3)
strides = (1, 1)
padding = (1, 1)
dtype = "float32"


def tune_kernels(tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"):
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = 2000
        length = len(task.config_space)
        print("length=", length)
        tuner_obj.tune(n_trial=min(n_trial, length),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


def tune_and_evaluate(tuning_opt, number, tune=True):
    op, params, data_shape = get_workload(batch_size, image_shape, out_channel, kernel_size, strides, padding)
    tasks = autotvm.task.extract_from_program(op, target=target, params=params, ops=(relay.op.nn.conv2d,))
    log_file = tuning_opt["log_filename"]
    if tune:
        print("Tuning...")
        tune_kernels(tasks, **tuning_opt)

    if not os.path.exists(log_file):
        raise RuntimeError("the log file {} doesn't exists".format(log_file))
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(op, target=target, params=params)

        ctx = tvm.context(str(target), 0)
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input("data", data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=number)
        prof_res = np.array(ftimer().results) * 1e3
        print("Time cost is: ", np.mean(prof_res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tune", help="tune mode", action="store_true")
    parser.add_argument("-n", "--number", type=int, default=100)
    parser.add_argument("-f", "--file", type=str, default="tune_log.log")
    parser.add_argument("-c", "--choice", type=str, default="gridsearch")
    args = parser.parse_args()
    tuning_option = {
        "log_filename": args.file,
        "tuner": args.choice,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
            # runner=autotvm.RPCRunner(
            #     '1080ti',  # change the device key to your key
            #     '0.0.0.0', 9190,
            #     number=20, repeat=3, timeout=4, min_repeat_ms=150)
        ),
    }
    tune_and_evaluate(tuning_option, args.number, args.tune)
