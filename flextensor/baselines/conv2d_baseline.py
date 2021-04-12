import os
import logging
import sys
import time
import argparse
import timeit
import torch
import tvm 
import topi
import numpy as np
import tvm.contrib.graph_runtime as runtime

from tvm import relay
from tvm.relay import testing
from tvm.relay.testing.init import create_workload
from tvm.relay.testing import layers
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from topi.testing import conv2d_nchw_python
from collections import namedtuple
from topi.util import get_const_tuple
from flextensor.configs.conv2d_config import *

torch.backends.cudnn.enabled = True


shape_dict = {
    "yolo": yolo_shapes,
    "google": google_shapes,
    "squeeze": squeeze_shapes,
    "res": res_shapes,
    "vgg-16": vgg_16_shapes,
    "vgg-19": vgg_19_shapes,
    "test": test_conv_shapes,
    "yolo_b8": yolo_shapes_b8,
    "mobile_v2": mobilev2_shapes,
}


def pytorch_cpu(batch_size, height, width, channel, kernel_size, output_channel, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
    run_time = timeit.timeit(setup= 'import torch\n'
                                    'conv = torch.nn.functional.conv2d\n'
                                    'A = torch.rand([' + str(batch_size) + ', ' + str(channel) + ', ' + str(height) + ', ' + str(width) + '], dtype=torch.float32)\n'
                                    'W = torch.rand([' + str(output_channel) + ', ' + str(channel//groups) + ', ' + str(kernel_size) + ', ' + str(kernel_size) + '], dtype=torch.float32)\n'
                                    'conv(A, W, stride=' + str(stride) + ', padding=' + str(padding) + ', dilation=' + str(dilation) + ', groups=' + str(groups) + ')\n',
                               stmt='ans = conv(A, W, stride=' + str(stride) + ', padding=' + str(padding) + ', dilation=' + str(dilation) + ', groups=' + str(groups) + ')',
                               number=number)
    return run_time / number * 1e3


def pytorch_cuda(N, H, W, C, kernel_size, K, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
    A = torch.rand([N, C, H, W], dtype=torch.float32).cuda("cuda:" + str(dev))
    W = torch.rand([K, C//groups, kernel_size, kernel_size], dtype=torch.float32).cuda("cuda:" + str(dev))

    # warm-up
    torch.nn.functional.conv2d(A, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    torch.cuda.synchronize()
    sum_time = 0.0
    for i in range(number):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ans = torch.nn.functional.conv2d(A, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        sum_time += start.elapsed_time(end)
    return sum_time / number


def tvm_generic_cuda(timeout=4, trials=100):
    def _inner(N, H, W, C, kernel_size, K, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
        return tvm_generic(N, H, W, C, kernel_size, K, stride=stride, padding=padding, dilation=dilation, groups=groups, 
                number=number, dev=dev, timeout=timeout, target="cuda", trials=trials)
    return _inner


def tvm_generic_llvm(timeout=4, trials=100):
    def _inner(N, H, W, C, kernel_size, K, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
        return tvm_generic(N, H, W, C, kernel_size, K, stride=stride, padding=padding, dilation=dilation, groups=groups, 
                number=number, dev=dev, timeout=timeout, target="llvm", trials=trials)
    return _inner


def tvm_generic(N, H, W, C, kernel_size, K, stride=1, padding=0, dilation=1, groups=1, 
    number=100, dev=0, timeout=4, target="llvm", trials=100):
    data_shape = (N, C, H, W)
    data = relay.var("data", shape=data_shape, dtype="float32")
    kernel_size = (kernel_size, kernel_size)
    stride = (stride, stride)
    padding = (padding, padding)
    body = layers.conv2d(data=data, channels=K, kernel_size=kernel_size, strides=stride, padding=padding, name="conv2d")
    op = relay.Function(relay.ir_pass.free_vars(body), body)
    sym, params = create_workload(op)
    tasks = autotvm.task.extract_from_program(op, target=target, params=params, ops=(relay.op.nn.conv2d,))
    tuning_option = {
        "log_filename": "tvm_baseline_{}.log".format((N, C, H, W, K, kernel_size, stride, padding, dilation, groups)),
        "tuner": "xgb",
        "early_stopping": 30,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=timeout),
            runner=autotvm.LocalRunner(number=number, repeat=1, timeout=timeout, min_repeat_ms=150),
            # runner=autotvm.RPCRunner(
            #     '1080ti',  # change the device key to your key
            #     '0.0.0.0', 9190,
            #     number=20, repeat=3, timeout=4, min_repeat_ms=150)
        ),
    }
    log_filename = tuning_option["log_filename"]
    tuner = tuning_option["tuner"]
    early_stopping = tuning_option["early_stopping"]
    measure_option = tuning_option["measure_option"]

    # only support one task
    assert len(tasks) == 1

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
        n_trial = trials
        length = len(task.config_space)
        print("config space length=", length)
        # tuner_obj.tune(n_trial=min(n_trial, length),
        #                early_stopping=early_stopping,
        #                measure_option=measure_option,
        #                callbacks=[
        #                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
        #                    autotvm.callback.log_to_file(log_filename)])
    
    if not os.path.exists(log_filename):
        raise RuntimeError("the log file {} doesn't exists".format(log_filename))
    with autotvm.apply_history_best(log_filename):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(op, target=target, params=params)

        ctx = tvm.device(str(target), 0)
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype("float32"))
        module = runtime.create(graph, lib, ctx)
        module.set_input("data", data_tvm)
        module.set_input(**params)

        # evaluate
        ftimer = module.module.time_evaluator("run", ctx, number=number, repeat=1)
        prof_res = np.array(ftimer().results) * 1e3
        return prof_res


def schedule_direct_cuda(cfg, s, conv):
    """schedule optimized for batch size = 1"""

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.current_target()
    if target.target_name in ['nvptx', 'rocm']:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    # fallback support
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            target.target_name, target.model, 'conv2d', 'direct')
        cfg.fallback_with_reference_log(ref_log)
    ##### space definition end #####

    pad_data, kernel = s[conv].op.input_tensors

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.te.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
        s[kernel].compute_inline()

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, 'local')
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope('local')
        OL = conv

    # create cache stage
    AA = s.cache_read(pad_data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, tvm.te.thread_axis("blockIdx.z"))
    s[output].bind(by, tvm.te.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.te.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.te.thread_axis("vthread"))
    s[output].bind(vy, tvm.te.thread_axis("vthread"))
    s[output].bind(vx, tvm.te.thread_axis("vthread"))
    s[output].bind(tf, tvm.te.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    ryo, ryi = cfg['tile_rx'].apply(s, OL, ry)
    rxo, rxi = cfg['tile_ry'].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, tvm.te.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.te.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.te.thread_axis("threadIdx.x"))

    # unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    N, CO, OH, OW = get_const_tuple(output.shape)
    _, KH, KW, CI = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW)


@autotvm.template
def conv2d_nchw(N, H, W, CO, CI, KH, KW, stride, padding, dilation):
    # assert N == 1, "Only consider batch_size = 1 in this template"

    data = tvm.te.placeholder((N, CI, H, W), name='data')
    kernel = tvm.te.placeholder((CO, CI, KH, KW), name='kernel')
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=dilation, out_dtype='float32')
    s = tvm.te.create_schedule([conv.op])

    cfg = autotvm.get_config()

    ##### space definition begin #####
    schedule_direct_cuda(cfg, s, conv)

    return s, [data, kernel, conv]


logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


def run(name, N, H, W, CO, CI, KH, KW, stride, pad, dilation, trials=100, timeout=4, number=10, target="llvm", dev=0, tune=True):
    N, H, W, CO, CI, KH, KW, strides, padding = N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad)
    task = autotvm.task.create(conv2d_nchw,
                               args=(N, H, W, CO, CI, KH, KW, strides, padding, dilation),
                               target=target)
    print("config_space length:", len(task.config_space))
    logfile = "conv2d_" + name + "_{}".format((N, CI, H, W, CO, KH, KW, stride, pad, dilation)) + ".log"

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=number, repeat=1, min_repeat_ms=150, timeout=timeout)
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    tuner = autotvm.tuner.XGBTuner(task)
    beg = time.time()
    print("Tune: ", tune)
    if tune:
        tuner.tune(n_trial=trials,
                measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file(logfile)])
    end = time.time()

    #########################################################################
    # Finally we can inspect the best config from log file, check correctness,
    # and measure running time.

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(logfile)
    best_config = dispatch_context.query(task.target, task.workload)
    print("Optimize use ", end - beg, "s")
    print("\nBest config:")
    print(best_config)

    # apply history best from log file
    with autotvm.apply_history_best(logfile):
        with tvm.target.create(target):
            s, arg_bufs = conv2d_nchw(N, H, W, CO, CI, KH, KW, strides, padding, dilation)
            # print(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs, "cuda")
            print(func.imported_modules[0].get_source())
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    # c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    ctx = tvm.device(str(target), dev)
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty((N, CO, (H + 2 * pad - dilation * (KH - 1) - 1) // stride + 1, (W + 2 * pad - dilation * (KW - 1) - 1) // stride + 1), ctx=ctx)
    # func(a_tvm, w_tvm, c_tvm)

    # tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
    cost = evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3
    return cost


def tvm_opt_cuda(name, trials=100, timeout=4, tune=True):
    def _inner(N, H, W, C, kernel_size, K, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
        return run(name, N, H, W, K, C, kernel_size, kernel_size, stride, padding, dilation, 
            trials=trials, timeout=timeout, number=number, target="cuda", dev=dev, tune=tune)
    return _inner


def tvm_opt_llvm(name, trials=100, timeout=4, tune=True):
    def _inner(N, H, W, C, kernel_size, K, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
        return run(name, N, H, W, K, C, kernel_size, kernel_size, stride, padding, dilation, 
            trials=trials, timeout=timeout, number=number, target="llvm", dev=dev, tune=tune)
    return _inner



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shapes", help="Use which shapes [yolo, google, res, squeeze, vgg-16, vgg-19]", type=str, default="yolo")
    parser.add_argument("-f", "--from_", help="From which shape", type=int, default=0)
    parser.add_argument("-t", "--to", help="To which shape", type=int, default=-1)
    parser.add_argument("-n", "--number", help="number test run", type=int, default=10)
    parser.add_argument("--target", help="target device type", type=str, default="llvm")
    parser.add_argument("--device", help="target device number", type=int, default=0)
    parser.add_argument("--type", help="type of baseline", type=str, default="pytorch")
    parser.add_argument("--timeout", help="timeout of baseline", type=int, default=4)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--batch", type=int, default=1, help="Specify the batch size of the shape")

    args = parser.parse_args()
    shapes = shape_dict[args.shapes]
    if args.to < 0:
        end = len(shapes)
    else:
        end = args.to
    shapes = shapes[args.from_:end]
    print("Changing batch size to ", args.batch)
    for i in range(len(shapes)):
        shapes[i] = list(shapes[i])
        shapes[i][0] = args.batch
        shapes[i] = tuple(shapes[i])

    if args.type == "pytorch":
        if args.target == "cuda":
            baseline = pytorch_cuda
        elif args.target == "llvm":
            baseline = pytorch_cpu
        else:
            raise RuntimeError("Only support target 'llvm' and 'cuda', but got %s"%args.target)
    elif args.type == "tvm_generic":
        if args.target == "cuda":
            baseline = tvm_generic_cuda(timeout=args.timeout, trials=args.trials)
        elif args.target == "llvm":
            baseline = tvm_generic_llvm(timeout=args.timeout, trials=args.trials)
        else:
            raise RuntimeError("Only support target 'llvm' and 'cuda', but got %s"%args.target)
    elif args.type == "tvm_opt":
        if args.target == "cuda":
            baseline = tvm_opt_cuda(args.shapes, timeout=args.timeout, trials=args.trials, tune=args.tune)
        elif args.target == "llvm":
            baseline = tvm_opt_llvm(args.shapes, timeout=args.timeout, trials=args.trials, tune=args.tune)
        else:
            raise RuntimeError("Only support target 'llvm' and 'cuda', but got %s"%args.target)
    else:
        raise RuntimeError("Only implement pytorch baseline now, no '%s' baseline"%args.type)
    
    print("%s baselines for %s convolution 2d for target %s (%d):" % (args.type, args.shapes, args.target, args.device))
    for i, shape in enumerate(shapes):
        count = i + args.from_ 
        print("layer", count, shape)
        batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
        batch = args.batch
        cost = baseline(batch, height, width, in_channel, k_h, out_channel, stride=stride, padding=padding, dilation=dilation, groups=groups, number=args.number, dev=args.device)
        print("Use %f(ms)" % cost)
        print()
    print("Done!")
