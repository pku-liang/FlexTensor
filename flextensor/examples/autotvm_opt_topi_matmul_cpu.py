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


def get_operator(data_shape, weight_shape, dtype="float32"):
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=dtype)
    body = relay.nn.dense(data=data, weight=weight, name="dense")
    return relay.Function(relay.ir_pass.free_vars(body), body)


def get_workload(data_shape, weight_shape, dtype="float32"):
    op = get_operator(data_shape, weight_shape, dtype=dtype)
    sym, params = create_workload(op)
    return sym, params, data_shape


target = "llvm"
batch_size = 256
image_shape = (1024, 7, 7)
out_channel = 1024
kernel_size = (3, 3)
strides = (1, 1)
padding = (1, 1)
dtype = "float32"

num_threads = 4
os.environ["TVM_NUM_THREADS"] = str(num_threads)


def tune_kernels(tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log", mode=0):
    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        name = "topi_nn_conv2d" if mode == 0 else "topi_x86_conv2d_NCHWc"
        print(name)
        task = autotvm.task.create(name, args=tsk.args, target=target, template_key="direct")
        task.workload = tsk.workload

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = min(len(task.config_space), 1000)
        print("trials=", n_trial)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


def tune_and_evaluate(tuning_opt, number, tune=True):
    op, params, data_shape = get_workload(batch_size, image_shape, out_channel, kernel_size, strides, padding)
    print(tvm.autotvm.task.task.TASK_TABLE)
    tasks = autotvm.task.extract_from_program(op, target=target, params=params, ops=(relay.op.nn.dense,))
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

        ctx = tvm.cpu(np.random.randint(0, 20))
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input("data", data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=number, repeat=1)
        prof_res = np.array(ftimer().results) * 1e3
        cost = np.mean(prof_res)
        print("Time cost is: ", cost)
    return cost


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--tune", help="tune mode", action="store_true")
    # parser.add_argument("-n", "--number", type=int, default=100)
    # parser.add_argument("-o", "--optimize", action="store_true")
    # parser.add_argument("-f", "--file", type=str, default="tune_log.log")
    # parser.add_argument("-c", "--choice", type=str, default="gridsearch")
    # args = parser.parse_args()
    arg_lst = [
        (1, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (8, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (64, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (256, 7, 7, 1024, 3, 3, 1024, 1, 1),
        (1, 14, 14, 1024, 1, 1, 512, 1, 0),
        (1, 28, 28, 256, 3, 3, 512, 1, 1),
        (1, 28, 28, 512, 1, 1, 256, 1, 0),
        (1, 56, 56, 128, 3, 3, 256, 1, 1),
        (1, 56, 56, 192, 1, 1, 128, 1, 0),
        (1, 112, 112, 64, 3, 3, 192, 1, 1),
        (1, 448, 448, 3, 7, 7, 64, 2, 3)
    ]
    names = [
        "yolo24_b1",
        # "yolo24_b8",
        # "yolo24_b64",
        # "yolo24_b256",
        "yolo19_b1",
        "yolo10_b1",
        "yolo7_b1",
        "yolo4_b1",
        "yolo3_b1",
        "yolo2_b1",
        "yolo1_b1"
    ]
    for i in range(len(arg_lst)):
        batch_size = arg_lst[i][0]
        image_shape = (arg_lst[i][3], arg_lst[i][1], arg_lst[i][2])
        out_channel = arg_lst[i][6]
        kernel_size = (arg_lst[i][4], arg_lst[i][5])
        strides = (arg_lst[i][7], arg_lst[i][7])
        padding = (arg_lst[i][8], arg_lst[i][8])
        logfile = "../logs/tune_conv_" + names[i] + "_cpu_test.log"
        tuning_option = {
            "log_filename": logfile,
            "tuner": "xgb",
            "early_stopping": None,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000)
            ),
            "mode": 0
        }
        cost = tune_and_evaluate(tuning_option, 10, True)
        with open("logs_autotvm_conv_cpu_test.log", "a") as f:
            f.write("{}{} cost {}\n".format(names[i], arg_lst[i], cost))
