import argparse
import timeit
import torch
import time
import tvm
import topi
import numpy as np
import logging
import sys
from auto_schedule.configs.PixelCNN_config import PixelCNN_shape
from auto_schedule.task import pixelcnn

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

torch.backends.cudnn.enabled = True


def pytorch_cpu(B, H, W, in_C, out_C, kernel_height, kernel_width, mask_type, bias, dilation, stride, padding, number=10, dev=0):
    Input = torch.rand([B, in_C, H, W], dtype=torch.float32)
    Kernel = torch.ones([out_C, in_C, kernel_height, kernel_width], dtype=torch.float32)
    assert mask_type in ['A', 'B']

    if mask_type == 'A':
        Kernel[:, :, kernel_height // 2, kernel_width // 2:] = 0
        Kernel[:, :, kernel_height // 2 + 1:, :] = 0
    elif mask_type == 'B':
        Kernel[:, :, kernel_height // 2, kernel_width // 2 + 1:] = 0
        Kernel[:, :, kernel_height // 2 + 1:, :] = 0

    conv = torch.nn.Conv2d(in_C, out_C, (kernel_height, kernel_width), stride=stride, dilation=dilation, padding=padding, bias=bias)
    if bias is not None:
        conv.bias = bias
    conv.weight.data = Kernel

    begin_time = time.time()
    for i in range(number):
        output = conv(Input)
    end_time = time.time()

    # ms
    return (end_time - begin_time) * 1e3 / number


def pytorch_cuda(B, H, W, C, out_C, kernel_height, kernel_width, mask_type, bias, dilation, stride, padding, number=10, dev=0):
    Input = torch.rand([B, C, H, W], dtype=torch.float32).cuda("cuda:" + str(dev))
    Kernel = torch.ones([out_C, C, kernel_height, kernel_width], dtype=torch.float32).cuda("cuda:" + str(dev))
    assert mask_type in ['A', 'B']

    if mask_type == 'A':
        Kernel[:, :, kernel_height // 2, kernel_width // 2:] = 0
        Kernel[:, :, kernel_height // 2 + 1:, :] = 0
    elif mask_type == 'B':
        Kernel[:, :, kernel_height // 2, kernel_width // 2 + 1:] = 0
        Kernel[:, :, kernel_height // 2 + 1:, :] = 0

    conv = torch.nn.Conv2d(C, out_C, (kernel_height, kernel_width), stride=stride, dilation=dilation, padding=padding, bias=bias)
    conv = conv.cuda("cuda:" + str(dev))
    if bias is not None:
        conv.bias = bias
    conv.weight.data = Kernel

    begin_time = time.time()
    for i in range(number):
        output = conv(Input)
    end_time = time.time()

    # ms
    return (end_time - begin_time) * 1e3 / number


def tvm_PixelCNN_cpu(B, H, W, C, out_C, kernel_height, kernel_width, mask_type, bias, dilation, stride, padding, number=10, dev=0):
    Input = torch.rand([B, H, W, C], dtype=torch.float32)
    Kernel = torch.zeros([out_C, C, kernel_height, kernel_width], dtype=torch.float32)

    s, bufs = pixelcnn(B, H, W, C, out_C, kernel_height, kernel_width, mask_type, bias, dilation=dilation, stride=stride, padding=padding)
    ctx = tvm.cpu(dev_id=dev)
    s =  tvm.create_schedule(s)
    f = tvm.build(s, bufs, "llvm")

    im = tvm.nd.array(Input.numpy().astype(np.float32), ctx)
    fi = tvm.nd.array(Kernel.numpy().astype(np.float32), ctx)

    in_height = H
    in_width = W
    out_height = (H + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (W + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    output_shape = (B, out_height, out_width, out_C)
    un = tvm.nd.array(np.zeros(output_shape).astype(np.float32), ctx)

    start_time = time.time()
    for i in range(number):
        f(im, fi, un)
    end_time = time.time()
    return (end_time - start_time) * 1e3 / number


def schedule_direct_cuda(cfg, s, mask, conv):
    """schedule optimized for batch size = 1"""

    ##### space definition begin #####
    bm, bo, bh, bw = s[mask].op.axis
    cfg.define_split("tile_bo", bo, num_outputs=4)
    cfg.define_split("tile_bh", bh, num_outputs=4)
    cfg.define_split("tile_bw", bw, num_outputs=4)
    bo, vo, to, oi= cfg["tile_bo"].apply(s, mask, bo)
    bh, vh, th, hi = cfg["tile_bh"].apply(s, mask, bh)
    bw, vw, tw, wi = cfg["tile_bw"].apply(s, mask, bw)
    s[mask].reorder(bo, bh, bw, vo, vh, vw, to, th, tw, oi, hi, wi)
    s[mask].bind(bo, tvm.thread_axis("blockIdx.z"))
    s[mask].bind(bh, tvm.thread_axis("blockIdx.y"))
    s[mask].bind(bw, tvm.thread_axis("blockIdx.x"))
    s[mask].bind(vo, tvm.thread_axis("vthread"))
    s[mask].bind(vh, tvm.thread_axis("vthread"))
    s[mask].bind(vw, tvm.thread_axis("vthread"))
    s[mask].bind(to, tvm.thread_axis("threadIdx.z"))
    s[mask].bind(th, tvm.thread_axis("threadIdx.y"))
    s[mask].bind(tw, tvm.thread_axis("threadIdx.x"))
    s[mask].bind(oi, tvm.thread_axis("blockIdx.z"))
    s[mask].bind(hi, tvm.thread_axis("blockIdx.y"))
    s[mask].bind(wi, tvm.thread_axis("blockIdx.x"))


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
    if isinstance(kernel.op, tvm.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
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
    s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
    s[output].bind(by, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
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
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    # unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    N, CO, OH, OW = get_const_tuple(output.shape)
    _, KH, KW, CI = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW)

@autotvm.template
def pixelcnn_autotvm(N, H, W, CO, CI, KH, KW, mask_type, bias, stride, padding, dilation):
    # assert N == 1, "Only consider batch_size = 1 in this template"

    # data = tvm.placeholder((N, CI, H, W), name='data')
    # kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')
    # conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=dilation, out_dtype='float32')
    convop, tensors = pixelcnn(N, H, W, CI, CO, KH, KW, mask_type, bias=bias, stride=stride, padding=padding, dilation=dilation)
    s = tvm.create_schedule(convop)

    cfg = autotvm.get_config()

    ##### space definition begin #####
    schedule_direct_cuda(cfg, s, *tensors[-2:])

    return s, [*tensors]


logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


def run(name, N, H, W, CO, CI, KH, KW, mask_type, bias, stride, pad, dilation, trials=100, timeout=4, number=10, target="llvm", dev=0, tune=True):
    N, H, W, CO, CI, KH, KW, strides, padding = N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad)
    task = autotvm.task.create(pixelcnn_autotvm,
                               args=(N, H, W, CO, CI, KH, KW, mask_type, bias, strides, padding, dilation),
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
            s, arg_bufs = pixelcnn_autotvm(N, H, W, CO, CI, KH, KW, mask_type, bias, strides, padding, dilation)
            # print(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs, "cuda")
            print(func.imported_modules[0].get_source())
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    # c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    ctx = tvm.context(str(target), dev)
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
    def _inner(N, H, W, C, kernel_size, K, mask_type, bias, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
        return run(name, N, H, W, K, C, kernel_size, kernel_size, mask_type, bias, stride, padding, dilation, 
            trials=trials, timeout=timeout, number=number, target="cuda", dev=dev, tune=tune)
    return _inner


def tvm_opt_llvm(name, trials=100, timeout=4, tune=True):
    def _inner(N, H, W, C, kernel_size, K, stride=1, padding=0, dilation=1, groups=1, number=100, dev=0):
        return run(name, N, H, W, K, C, kernel_size, kernel_size, stride, padding, dilation, 
            trials=trials, timeout=timeout, number=number, target="llvm", dev=dev, tune=tune)
    return _inner

if __name__ == "__main__":
    shapes = PixelCNN_shape

    parser = argparse.ArgumentParser()
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
        print("Warm up...")
        baseline(*shapes[0]);
        for shape in shapes:
            print("Shape {} on target {}".format(shape, args.target))
            cost = baseline(*shape, number=args.number, dev=args.device)
            print("Pytorch cost on {}: {}ms".format(args.target, cost))
    elif args.type == "tvm_opt":
        if args.target == "cuda":
            baseline = tvm_opt_cuda("pixelCNN", timeout=args.timeout, trials=args.trials, tune=args.tune)
        elif args.target == "llvm":
            baseline = tvm_opt_llvm("pixelCNN", timeout=args.timeout, trials=args.trials, tune=args.tune)
        else:
            raise RuntimeError("Only support target 'llvm' and 'cuda', but got %s"%args.target)
        for shape in shapes:
            print("Shape {} on target {}".format(shape, args.target))
            batch, height, width, in_channel, out_channel, k_h, k_w, mask_type, bias, dilation, stride, padding = shape
            cost = baseline(batch, height, width, in_channel, k_h, out_channel, mask_type, bias, stride=stride, padding=padding, dilation=dilation)
            print("Pytorch cost on {}: {}ms".format(args.target, cost))
    else:
        raise RuntimeError("Only implement pytorch baseline now, no '%s' baseline"%args.type)
        
    print("Done!")
