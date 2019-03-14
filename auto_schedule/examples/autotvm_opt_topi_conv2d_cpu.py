import tvm
import os
import numpy as np
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


target = "llvm"
batch_size = 1
image_shape = (256, 14, 14)
out_channel = 512
kernel_size = (3, 3)
strides = (1, 1)
padding = (1, 1)
dtype = "float32"
log_file = "topi_conv2d.log"

num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000)
    )
}


def tune_kernels(tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"):
    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        op_name = tsk.workload[0]
        task = autotvm.task.create("topi_nn_conv2d", args=tsk.args, target=target, template_key="direct")
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
        n_trial = len(task.config_space)
        print("trials=", n_trial)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


def tune_and_evaluate(tuning_opt):
    op, params, data_shape = get_workload(batch_size, image_shape, out_channel, kernel_size, strides, padding)
    tasks = autotvm.task.extract_from_program(op, target=target, params=params, ops=(relay.op.nn.conv2d,))

    print("Tuning...")
    tune_kernels(tasks, **tuning_opt)

    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(op, target=target, params=params)

        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input("data", data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=1)
        prof_res = np.array(ftimer().results) * 1e3
        print("Time cost is: ", np.mean(prof_res))


tune_and_evaluate(tuning_option)
