import argparse
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tvm
import topi
import numpy as np
import logging
import sys
from auto_schedule.testing.configs.gated_pixelcnn_config import gated_pixelcnn_shape
from auto_schedule.testing.task import gatedpixelcnn

from tvm import relay
from tvm.relay import testing
from tvm.relay.testing.init import create_workload
from tvm.relay.testing import layers
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from collections import namedtuple
from topi.util import get_const_tuple

torch.backends.cudnn.enabled = True

def same_padding(kernel_size):
    # assumming stride 1
    if isinstance(kernel_size, int):
        return kernel_size // 2
    else:
        return (kernel_size[0] // 2, kernel_size[1] // 2)

class MaskedConvolution2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
            *args, mask='B', vertical=False, mask_mode="noblind", **kwargs):
        if "padding" not in kwargs:
            assert "stride" not in kwargs
            kwargs["padding"] = same_padding(kernel_size)
        remove = {"conditional_features", "conditional_image_channels"}
        for feature in remove:
            if feature in kwargs:
                del kwargs[feature]
        super(MaskedConvolution2D, self).__init__(in_channels,
                out_channels, kernel_size, *args, **kwargs)
        Cout, Cin, kh, kw = self.weight.size()
        pre_mask = np.ones_like(self.weight.data.cpu().numpy()).astype(np.float32)
        yc, xc = kh // 2, kw // 2

        assert mask_mode in {"noblind", "turukin", "fig1-van-den-oord"}
        if mask_mode == "noblind":
            # context masking - subsequent pixels won't hav access
            # to next pixels (spatial dim)
            if vertical:
                if mask == 'A':
                    # In the first layer, can ONLY access pixels above it
                    pre_mask[:, :, yc:, :] = 0.0
                else:
                    # In the second layer, can access pixels above or even with it.
                    # Reason being that the pixels to the right or left of the current pixel
                    #  only have a receptive field of the layer above the current layer and up.
                    pre_mask[:, :, yc+1:, :] = 0.0
            else:
                # All rows after center must be zero
                pre_mask[:, :, yc+1:, :] = 0.0
                ### All rows before center must be zero # XXX: not actually necessary
                ##pre_mask[:, :, :yc, :] = 0.0
                # All columns after center in center row must be zero
                pre_mask[:, :, yc, xc+1:] = 0.0

            if mask == 'A':
                # Center must be zero in first layer
                pre_mask[:, :, yc, xc] = 0.0
            # same pixel masking - pixel won't access next color (conv filter dim)
            #def bmask(i_out, i_in):
            #    cout_idx = np.expand_dims(np.arange(Cout) % 3 == i_out, 1)
            #    cin_idx = np.expand_dims(np.arange(Cin) % 3 == i_in, 0)
            #    a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
            #    return a1 * a2

            #for j in range(3):
            #    pre_mask[bmask(j, j), yc, xc] = 0.0 if mask == 'A' else 1.0

            #pre_mask[bmask(0, 1), yc, xc] = 0.0
            #pre_mask[bmask(0, 2), yc, xc] = 0.0
            #pre_mask[bmask(1, 2), yc, xc] = 0.0
        elif mask_mode == "fig1-van-den-oord":
            if vertical:
                pre_mask[:, :, yc:, :] = 0.0
            else:
                # All rows after center must be zero
                pre_mask[:, :, yc+1:, :] = 0.0
                ### All rows before center must be zero # XXX: not actually necessary
                ##pre_mask[:, :, :yc, :] = 0.0
                # All columns after center in center row must be zero
                pre_mask[:, :, yc, xc+1:] = 0.0

            if mask == 'A':
                # Center must be zero in first layer
                pre_mask[:, :, yc, xc] = 0.0
        elif mask_mode == "turukin":
            pre_mask[:, :, yc+1:, :] = 0.0
            pre_mask[:, :, yc, xc+1:] = 0.0
            if mask == 'A':
                pre_mask[:, :, yc, xc] = 0.0

        print("%s %s MASKED CONV: %d x %d. Mask:" % (mask, "VERTICAL" if vertical else "HORIZONTAL", kh, kw))
        print(pre_mask[0, 0, :, :])

        self.register_buffer("mask", torch.from_numpy(pre_mask))

    def __call__(self, x):
        self.weight.data = self.weight.data * self.mask
        return super(MaskedConvolution2D, self).forward(x)

class PixelCNNGatedLayer(nn.Module):
    def __init__(self, primary, in_channels, out_channels, filter_size,
            mask='B', nobias=False, conditional_features=None,
            conditional_image_channels=None, residual_vertical=False,
            residual_horizontal=False, skips=False, gated=True,
            relu_out=False, horizontal_2d_convs=False, mask_mode="noblind"):
        super().__init__()
        self.primary = primary
        if primary:
            assert mask == 'A'
            assert not residual_vertical
            assert not residual_horizontal
        else:
            assert mask == 'B'
        self.out_channels = out_channels
        self.gated = gated
        gm = 2 if gated else 1
        self.vertical_conv = MaskedConvolution2D(
            in_channels, gm * out_channels, (filter_size, filter_size),
            mask=mask, vertical=True, mask_mode=mask_mode)
        self.v_to_h_conv = nn.Conv2d(gm * out_channels, gm * out_channels, 1)

        self.horizontal_conv = MaskedConvolution2D(
            in_channels, gm * out_channels,
            (filter_size if horizontal_2d_convs else 1, filter_size), # XXX: traditionally (1, filter_size),
            mask=mask, vertical=False, mask_mode=mask_mode)

        self.residual_vertical = None
        if residual_vertical:
            self.residual_vertical = nn.Conv2d(in_channels, gm * out_channels, 1)

        self.horizontal_output = nn.Conv2d(out_channels, out_channels, 1)
        self.horizontal_skip = None
        if skips:
            self.horizontal_skip = nn.Conv2d(out_channels, out_channels, 1)
        self.conditional_vector = conditional_features is not None
        self.conditional_image = conditional_image_channels is not None
        if self.conditional_image:
            self.cond_conv_h = nn.Conv2d(conditional_image_channels, gm * out_channels, 1, bias=False)
            self.cond_conv_v = nn.Conv2d(conditional_image_channels, gm * out_channels, 1, bias=False)
        if self.conditional_vector:
            self.cond_fc_h = nn.Linear(conditional_features, gm * out_channels, bias=False)
            self.cond_fc_v = nn.Linear(conditional_features, gm * out_channels, bias=False)
        self.residual_horizontal = residual_horizontal
        self.relu_out = relu_out

    @classmethod
    def primary(cls, in_channels, out_channels, filter_size,
            nobias=False, conditional_features=None,
            conditional_image_channels=None, 
            skips=False, gated=True,
            relu_out=False, horizontal_2d_convs=False, mask_mode="noblind"):
        return cls(True, in_channels, out_channels, filter_size, nobias=nobias,
                mask='A', conditional_features=conditional_features,
                conditional_image_channels=conditional_image_channels,
                residual_vertical=False, residual_horizontal=False,
                skips=skips, gated=gated,
                relu_out=relu_out, horizontal_2d_convs=horizontal_2d_convs,
                mask_mode=mask_mode)
    
    @classmethod
    def secondary(cls, in_channels, out_channels, filter_size,
            nobias=False, conditional_features=None,
            conditional_image_channels=None, residual_vertical=True,
            residual_horizontal=True, skips=False, gated=True,
            relu_out=False, horizontal_2d_convs=False, mask_mode="noblind"):
        return cls(False, in_channels, out_channels, filter_size, nobias=nobias,
                mask='B', conditional_features=conditional_features,
                conditional_image_channels=conditional_image_channels,
                residual_vertical=residual_vertical, residual_horizontal=residual_horizontal,
                skips=skips, gated=gated, relu_out=relu_out,
                horizontal_2d_convs=horizontal_2d_convs, mask_mode=mask_mode)

    def _gate(self, x):
        if self.gated:
            return F.tanh(x[:,:self.out_channels]) * F.sigmoid(x[:,self.out_channels:])
        else:
            return x

    def __call__(self, v, h, conditional_image=None, conditional_vector=None):
        horizontal_preactivation = self.horizontal_conv(h) # 1xN
        vertical_preactivation = self.vertical_conv(v) # NxN
        v_to_h = self.v_to_h_conv(vertical_preactivation) # 1x1
        if self.residual_vertical is not None:
            vertical_preactivation = vertical_preactivation + self.residual_vertical(v) # 1x1 to residual
        horizontal_preactivation = horizontal_preactivation + v_to_h
        if self.conditional_image and conditional_image is not None:
            horizontal_preactivation = horizontal_preactivation + \
                    self.cond_conv_h(conditional_image)
            vertical_preactivation = vertical_preactivation + \
                    self.cond_conv_v(conditional_image)
        if self.conditional_vector and conditional_vector is not None:
            horizontal_preactivation = horizontal_preactivation + \
                    self.cond_fc_h(conditional_vector).unsqueeze(-1).unsqueeze(-1)
            vertical_preactivation = vertical_preactivation + \
                    self.cond_fc_v(conditional_vector).unsqueeze(-1).unsqueeze(-1)
        v_out = self._gate(vertical_preactivation)
        h_activated = self._gate(horizontal_preactivation)
        h_skip = None
        if self.horizontal_skip is not None:
            h_skip = self.horizontal_skip(h_activated)
        h_preres = self.horizontal_output(h_activated)
        if self.residual_horizontal:
            h_out = h + h_preres
        else:
            h_out = h_preres
        if self.relu_out:
            v_out = F.relu(v_out)
            h_out = F.relu(h_out)
            if h_skip is not None:
                h_skip = F.relu(h_skip)
        
        return v_out, h_out, h_skip

def pytorch_cpu(B, H, W, in_C, out_C, kernel_size, classVector, bias, dilation, stride, padding, number=10, dev=0):
    Input = torch.rand([B, in_C, H, W], dtype=torch.float32)

    gatedLayer = PixelCNNGatedLayer(primary=False, in_channels=in_C, out_channels=out_C, filter_size=kernel_size)

    begin_time = time.time()
    for i in range(number):
        output = gatedLayer(Input, Input)
    end_time = time.time()

    # ms
    return (end_time - begin_time) * 1e3 / number

def pytorch_cuda(B, H, W, in_C, out_C, kernel_size, classVector, bias, dilation, stride, padding, number=10, dev=0):
    Input = torch.rand([B, in_C, H, W], dtype=torch.float32).cuda("cuda:" + str(dev))

    gatedLayer = PixelCNNGatedLayer(primary=False, in_channels=in_C, out_channels=out_C, filter_size=kernel_size).cuda("cuda:" + str(dev))

    begin_time = time.time()
    for i in range(number):
        output = gatedLayer(Input, Input)
    end_time = time.time()

    # ms
    return (end_time - begin_time) * 1e3 / number

def tvm_GatedPixelCNN_cpu(B, H, W, in_C, out_C, kernel_size, classVector, bias, dilation, stride, padding, number=10, dev=0):
    Input = torch.rand([B, H, W, in_C], dtype=torch.float32)
    KernelV = torch.rand([2 * out_C, in_C, kernel_size, kernel_size], dtype=torch.float32)
    KernelV2H = torch.rand([2 * out_C, 2 * out_C, 1, 1], dtype=torch.float32)
    KernelH = torch.rand([2 * out_C, in_C, 1, kernel_size], dtype=torch.float32)
    KernelHOut = torch.rand([out_C, out_C, 1, 1], dtype=torch.float32)


    s, bufs = gatedpixelcnn(B, H, W, in_C, out_C, kernel_size, ClassVector=classVector, bias=bias, stride=stride, padding=padding, dilation=dilation)
    s =  tvm.create_schedule(s)
    ctx = tvm.cpu(dev)
    f = tvm.build(s, bufs, 'llvm')

    im = tvm.nd.array(Input.numpy().astype(np.float32), ctx)
    kv = tvm.nd.array(KernelV.numpy().astype(np.float32), ctx)
    kv2h = tvm.nd.array(KernelV2H.numpy().astype(np.float32), ctx)
    kh = tvm.nd.array(KernelH.numpy().astype(np.float32), ctx)
    kho = tvm.nd.array(KernelHOut.numpy().astype(np.float32), ctx)

    in_height = Input.shape[2]
    in_width = Input.shape[3]
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size
    tmp_output_shape = bufs[-1].shape
    output_shape = [int(x.value) for x in tmp_output_shape]
    un = tvm.nd.array(np.zeros(output_shape).astype(np.float32), ctx)

    start_time = time.time()
    for i in range(number):
        f(im, kv, kv2h, kh, kho, un)
    end_time = time.time()
    return (end_time - start_time) * 1e3 / number

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
def gatedPixelCNN(N, H, W, CO, CI, KH, ClassVector, bias, dilation, stride, padding):
    # assert N == 1, "Only consider batch_size = 1 in this template"
    outputs, inputs = gatedpixelcnn(N, H, W, CI, CO, KH, ClassVector=ClassVector, bias=bias, dilation=dilation, stride=stride, padding=padding)
    s = tvm.create_schedule(outputs)
    Input, KernelV, KernelV2H, KernelH, KernelHOut, GateV, Output = inputs

    cfg = autotvm.get_config()

    ##### space definition begin #####
    schedule_direct_cuda(cfg, s, outputs[-1])

    return s, inputs


logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


def run(name, N, H, W, CI, CO, KH, KW, ClassVector, bias, dilation, stride, padding, trials=100, timeout=4, number=10, target="llvm", dev=0, tune=True):
    task = autotvm.task.create(gatedPixelCNN,
                               args=(N, H, W, CO, CI, KH, ClassVector, bias, dilation, stride, padding),
                               target=target)
    print("config_space length:", len(task.config_space))
    logfile = "conv2d_" + name + "_{}".format((N, CI, H, W, CO, KH, dilation, stride, padding)) + ".log"

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
            s, arg_bufs = gatedPixelCNN(N, H, W, CO, CI, KH, KW, ClassVector, bias, stride, padding, dilation)
            # print(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs, "cuda")
            print(func.imported_modules[0].get_source())
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    # c_np = gatedPixelCNN_python(a_np, w_np, strides, padding)

    ctx = tvm.context(str(target), dev)
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty((N, CO, (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1, (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1), ctx=ctx)
    # func(a_tvm, w_tvm, c_tvm)

    # tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
    cost = evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3
    return cost


def tvm_opt_cuda(name, trials=100, timeout=4, tune=True):
    def _inner(N, H, W, CI, CO, KH, KW, ClassVector, bias, dilation, stride, padding, number=100, dev=0):
        return run(name, N, H, W, CI, CO, KH, KW, ClassVector, bias, dilation, stride, padding, 
            trials=trials, timeout=timeout, number=number, target="cuda", dev=dev, tune=tune)
    return _inner


def tvm_opt_llvm(name, trials=100, timeout=4, tune=True):
    def _inner(N, H, W, CI, CO, KH, KW, ClassVector, bias, dilation, stride, padding, number=100, dev=0):
        print("Run stride: ", stride)    
        return run(name, N, H, W, CI, CO, KH, KW, ClassVector, bias, dilation, stride, padding, 
            trials=trials, timeout=timeout, number=number, target="llvm", dev=dev, tune=tune)
    return _inner

if __name__ == "__main__":
    shapes = gated_pixelcnn_shape
    print("Not implemented yet...")
    sys.exit(0)

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
        baseline(*shapes[0], number=args.number, dev=args.device);
        for shape in shapes:
            print("Shape {} on target {}".format(shape, args.target))
            cost = baseline(*shapes[0], number=args.number, dev=args.device)
            print("Pytorch cost on {}: {}ms".format(args.target, cost))
    elif args.type == "tvm_opt":
        if args.target == "cuda":
            baseline = tvm_opt_cuda("gatedPixelCNN", timeout=args.timeout, trials=args.trials, tune=args.tune)
        elif args.target == "llvm":
            baseline = tvm_opt_llvm("gatedPixelCNN", timeout=args.timeout, trials=args.trials, tune=args.tune)
        else:
            raise RuntimeError("Only support target 'llvm' and 'cuda', but got %s"%args.target)
        for shape in shapes:
            print("Shape {} on target {}".format(shape, args.target))
            cost = baseline(*shape, number=args.number, dev=args.device)
            print("Autotvm cost on {}: {}ms".format(args.target, cost))

    else:
        raise RuntimeError("Only implement pytorch baseline now, no '%s' baseline"%args.type)
        
    print("Done!")