"""
Tuning High Performance Convolution on NVIDIA GPUs
=========================================================================
**Author**: `Lianmin Zheng <https://https://github.com/merrymercy>`_

This is an advanced tutorial for writing high performance tunable template for 
NVIDIA GPU. By running auto-tuner on this template, we can outperform the
vendor provided library CuDNN in many cases.
"""

######################################################################
# Install dependencies
# --------------------
# To use autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado
#
# To make tvm run faster in tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import logging
import sys
import numpy as np

import tvm
import topi
from topi.testing import conv2d_nchw_python
from collections import namedtuple

from tvm import autotvm
from topi.util import get_const_tuple

######################################################################
# Step 1:  Define the search space
# --------------------------------
# There are plenty of useful schedule primitives in tvm. You can also find 
# some tutorials that describe them in more details, such as 
# (1). :ref:`opt-conv-gpu`
# (2). `Optimizing DepthwiseConv on NVIDIA GPU <https://tvm.ai/2017/08/22/Optimize-Deep-Learning-GPU-Operators-with-TVM-A-Depthwise-Convolution-Example.html>`_
# 
# However, their implementations are manually tuned for some special input
# shapes. In this section, we build a large enough space to cover
# the techniques used in these tutorials. Then we rely on the efficient auto-tuner
# to search through this space and pick some good configurations.
# 
# If you are familiar with writing cuda schedule, you can find the following
# template is very general. Actually this template can be easily modified 
# to tune other operators such as depthwise convolution and gemm.
# In order to fully understand this template, you should be familiar with
# the schedule primitives and auto tuning API. You can refer to the above
# tutorials and :doc:`autotvm tutorial <tune_simple_template>`
#
# It is worth noting that the search space for a conv2d operator
# can be very large (at the level of 10^9 for some input shapes)
#

# to run this, first:
# export PATH=/usr/local/cuda-10.1/nvvm/libdevice:$PATH


def schedule_direct_cuda(cfg, s, conv):
    """schedule optimized for batch size = 1"""

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_n", n, num_outputs=4)
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    yx = s[conv].fuse(y, x)
    yx.dom = namedtuple("dom", ("extent"))
    yx.dom.extent = y.dom.extent.value * x.dom.extent.value
    print(yx.dom.extent)
    cfg.define_split("tile_yx", yx, num_outputs=4)

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
    yx = s[output].fuse(y, x)

    bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    byx, vyx, tyx, yxi = cfg["tile_yx"].apply(s, output, yx)


    s[output].bind(bn, tvm.thread_axis("blockIdx.z"))
    s[output].bind(bf, tvm.thread_axis("blockIdx.y"))
    s[output].bind(byx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vn, tvm.thread_axis("vthread"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vyx, tvm.thread_axis("vthread"))
    s[output].bind(tn, tvm.thread_axis("threadIdx.z"))
    s[output].bind(tf, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tyx, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(bn, bf, byx, vn, vf, vyx, tn, tf, tyx, ni, fi, yxi)
    s[OL].compute_at(s[output], tyx)

    # tile reduction axes
    n, f, yx = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    ryo, ryi = cfg['tile_rx'].apply(s, OL, ry)
    rxo, rxi = cfg['tile_ry'].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, yx)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_n"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_yx"].size[2])
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
def conv2d_nchw(N, H, W, CO, CI, KH, KW, stride, padding, dilation):
    # assert N == 1, "Only consider batch_size = 1 in this template"

    data = tvm.placeholder((N, CI, H, W), name='data')
    kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=dilation, out_dtype='float32')
    s = tvm.create_schedule([conv.op])

    cfg = autotvm.get_config()

    ##### space definition begin #####
    schedule_direct_cuda(cfg, s, conv)

    return s, [data, kernel, conv]

######################################################################
# Step 2:  Search through the space
# ---------------------------------
# We pick the last layer on resnet as test case.
# Since our space is very large, :code:`XGBoostTuner` is most suitable
# for our case. Here we only do 20 trials for demonstration.
# In practice, making 1000 trials usually can find some good kernels
# for this template

# logging config (for printing tuning log to screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


# the last layer in yolo
def run(name, N, H, W, CO, CI, KH, KW, stride, pad, dilation):
    N, H, W, CO, CI, KH, KW, strides, padding = N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad)
    task = autotvm.task.create(conv2d_nchw,
                               args=(N, H, W, CO, CI, KH, KW, strides, padding, dilation),
                               target='cuda')
    print(task.config_space)
    logfile = "conv2d_" + name + ".log"

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=10)
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=1000,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(logfile)])

    #########################################################################
    # Finally we can inspect the best config from log file, check correctness,
    # and measure running time.

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(logfile)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)

    # apply history best from log file
    with autotvm.apply_history_best(logfile):
        with tvm.target.create("cuda"):
            s, arg_bufs = conv2d_nchw(N, H, W, CO, CI, KH, KW, strides, padding, dilation)
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    # c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    ctx = tvm.gpu()
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty((N, CO, (H + 2 * pad - KH) // stride + 1, (W + 2 * pad - KW) // stride + 1), ctx=ctx)
    # func(a_tvm, w_tvm, c_tvm)

    # tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    cost = evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3
    print('Time cost of this operator: %f' % cost)
    with open("autotvm_conv_nchw.txt", "a") as f:
        f.write("name, {}\n".format(cost))


yolo_shapes = [
    # yolo
    # (1, 3, 448, 448, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    # (1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    # (1, 192, 56, 56, 128, 192, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    # (1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1),  # conv4   3
    # (1, 256, 56, 56, 256, 256, 1, 1, 1, 1, 0, 1, 1),  # conv5   4
    # (1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv7   6
    # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv8   7
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv9
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv10
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv11
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv12
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv13
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv14
    # (1, 512, 28, 28, 512, 512, 1, 1, 1, 1, 0, 1, 1),  # conv15      8
    # (1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv16     9
    # (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv17    10
    # (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv18     11
    # # (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv19
    # # (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv20
    # (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv21   12
    # (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1),  # conv22   13
    (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv23     14
    # (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv24
]


if __name__ == "__main__":
    arg_lst = yolo_shapes
    names = ["yolo_" + str(i) for i in range(len(arg_lst))]
    for i in range(len(arg_lst)):
        name = names[i]
        N, CI, H, W, CO, _, KH, KW, _, stride, pad, dilation, groups = arg_lst[i]
        N = 64
        run(name, N, H, W, CO, CI, KH, KW, stride, pad, dilation)

