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

from tvm import autotvm
from topi.util import get_const_tuple, traverse_inline, simplify
from topi.nn.util import get_pad_tuple

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


def _depthwise_conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Depthwise convolution nchw forward operator.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.Tensor
        4-D with shape [in_channel, channel_multiplier, filter_height, filter_width]

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str, optional
        Output data type

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    # shape of dilated kernel
    filter_channel, channel_multiplier, filter_height, filter_width = Filter.shape

    dilated_kernel_h = (filter_height - 1) * dilation_h + 1
    dilated_kernel_w = (filter_width - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = simplify(in_channel * channel_multiplier)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # padding stage
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    PaddedInput = topi.nn.pad(Input, pad_before, pad_after, name="PaddedInput")
    # depthconv stage
    di = tvm.te.reduce_axis((0, filter_height), name='di')
    dj = tvm.te.reduce_axis((0, filter_width), name='dj')
    Output = tvm.te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j: tvm.te.sum(
            (PaddedInput[b, c/channel_multiplier, i*stride_h+di*dilation_h,
                         j*stride_w+dj*dilation_w].astype(out_dtype) *
             Filter[c/channel_multiplier, c%channel_multiplier, di, dj].astype(out_dtype)),
            axis=[di, dj]),
        name='DepthwiseConv2d', tag="depthwise_conv2d_nchw")
    return Output


def schedule_depthwise_conv2d_nchw_cuda(cfg, s, outs):
    """Schedule for depthwise_conv2d nchw forward.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
    # s = tvm.te.create_schedule([x.op for x in outs])

    def _callback(op):
        pad_data = op.input_tensors[0]
        kernel = op.input_tensors[1]
        conv = op.output(0)

        ##### space definition begin #####
        n, f, y, x = s[conv].op.axis
        cfg.define_split("tile_f", f, num_outputs=4)
        cfg.define_split("tile_y", y, num_outputs=4)
        cfg.define_split("tile_x", x, num_outputs=4)
        cfg.define_knob("auto_unroll_max_step", [0, 256, 1500])

        # target = tvm.target.current_target()
        # if target.target_name in ['nvptx', 'rocm']:
        #     cfg.define_knob("unroll_explicit", [1])
        # else:
        #     cfg.define_knob("unroll_explicit", [0, 1])
        cfg.define_knob("unroll_explicit", [0, 1])

        # # fallback support
        # if cfg.is_fallback:
        #     ref_log = autotvm.tophub.load_reference_log(
        #         target.target_name, target.model, 'depthwise_conv2d_nchw', 'direct')
        #     cfg.fallback_with_reference_log(ref_log)
        #     # TODO(lmzheng): A bug here, set unroll_explicit to False as workaround
        #     cfg['unroll_explicit'].val = 0
        # ##### space definition end #####

        s[pad_data].compute_inline()
        # if isinstance(kernel.op, tvm.te.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
        #     s[kernel].compute_inline()

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
        AL = s.cache_read(AA, 'local', [OL])
        WL = s.cache_read(WW, 'local', [OL])

        # tile and bind spatial axes
        n, f, y, x = s[output].op.axis
        bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
        by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
        bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

        kernel_scope, n = s[output].split(n, nparts=1)
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

        # cooperative fetching
        s[AA].compute_at(s[output], bx)
        s[WW].compute_at(s[output], bx)
        s[AL].compute_at(s[output], tx)
        s[WL].compute_at(s[output], tx)

        for load in [AA, WW]:
            fused = s[load].fuse(*list(s[load].op.axis))
            fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
            fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
            fused, tz = s[load].split(fused, cfg["tile_f"].size[2])
            s[load].bind(tz, tvm.te.thread_axis("threadIdx.z"))
            s[load].bind(ty, tvm.te.thread_axis("threadIdx.y"))
            s[load].bind(tx, tvm.te.thread_axis("threadIdx.x"))

        s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
        s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    _callback(outs[0].op)



@autotvm.template
def depthwise_conv2d_nchw(N, H, W, factor, CI, KH, KW, stride, padding, dilation):
    data = tvm.te.placeholder((N, CI, H, W), name='data')
    kernel = tvm.te.placeholder((CI, factor, KH, KW), name='kernel')
    conv = _depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=dilation, out_dtype='float32')

    s = tvm.te.create_schedule([conv.op])
    cfg = autotvm.get_config()

    ##### space definition begin #####
    schedule_depthwise_conv2d_nchw_cuda(cfg, s, conv)

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
def run(name, N, H, W, factor, CI, KH, KW, strides, padding, dilation):
    # s, arg_bufs = depthwise_conv2d_nchw(N, H, W, factor, CI, KH, KW, strides, padding, dilation)
    task = autotvm.task.create(depthwise_conv2d_nchw,
                               args=(N, H, W, factor, CI, KH, KW, strides, padding, dilation),
                               target='cuda')
    print(task.config_space)
    logfile = "depthwise_" + name + ".log"

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
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
            s, arg_bufs = depthwise_conv2d_nchw(N, H, W, factor, CI, KH, KW, strides, padding, dilation)
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CI, factor, KH, KW)).astype(np.float32)
    # c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    ctx = tvm.gpu()
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty((N, factor * CI, (H + 2 * pad - KH) // stride + 1, (W + 2 * pad - KW) // stride + 1), ctx=ctx)
    # func(a_tvm, w_tvm, c_tvm)

    # tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    cost = evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3
    print('Time cost of this operator: %f' % cost)
    with open("autotvm_conv_nchw.txt", "a") as f:
        f.write("name, {}\n".format(cost))


mobilev2_shapes_b1 = [
    (1, 32, 112, 112, 32, 32, 3, 3, 1, 1, 1, 1, 32),
    (1, 16, 112, 112, 16 * 6, 16, 3, 3, 1, 2, 1, 1, 16),
    (1, 24, 56, 56, 24 * 6, 24, 3, 3, 1, 2, 1, 1, 24),
    (1, 32, 28, 28, 32 * 6, 32, 3, 3, 1, 2, 1, 1, 32),
    (1, 64, 14, 14, 64 * 6, 64, 3, 3, 1, 1, 1, 1, 64),
    (1, 96, 14, 14, 96 * 6, 96, 3, 3, 1, 2, 1, 1, 96),
    (1, 160, 7, 7, 160 * 6, 160, 3, 3, 1, 1, 1, 1, 160),
]


mobilev2_shapes_b8 = [
    (8, 32, 112, 112, 32, 32, 3, 3, 1, 1, 1, 1, 32),
    (8, 16, 112, 112, 16 * 6, 16, 3, 3, 1, 2, 1, 1, 16),
    (8, 24, 56, 56, 24 * 6, 24, 3, 3, 1, 2, 1, 1, 24),
    (8, 32, 28, 28, 32 * 6, 32, 3, 3, 1, 2, 1, 1, 32),
    (8, 64, 14, 14, 64 * 6, 64, 3, 3, 1, 1, 1, 1, 64),
    (8, 96, 14, 14, 96 * 6, 96, 3, 3, 1, 2, 1, 1, 96),
    (8, 160, 7, 7, 160 * 6, 160, 3, 3, 1, 1, 1, 1, 160),
]


mobilev2_shapes_b16 = [
    (16, 32, 112, 112, 32, 32, 3, 3, 1, 1, 1, 1, 32),
    (16, 16, 112, 112, 16 * 6, 16, 3, 3, 1, 2, 1, 1, 16),
    (16, 24, 56, 56, 24 * 6, 24, 3, 3, 1, 2, 1, 1, 24),
    (16, 32, 28, 28, 32 * 6, 32, 3, 3, 1, 2, 1, 1, 32),
    (16, 64, 14, 14, 64 * 6, 64, 3, 3, 1, 1, 1, 1, 64),
    (16, 96, 14, 14, 96 * 6, 96, 3, 3, 1, 2, 1, 1, 96),
    (16, 160, 7, 7, 160 * 6, 160, 3, 3, 1, 1, 1, 1, 160),
]


if __name__ == "__main__":
    arg_lst = mobilev2_shapes_b1
    names = ["mobile_" + str(i) for i in range(len(arg_lst))]
    for i in range(len(arg_lst)):
        name = names[i]
        N, CI, H, W, CO, _, KH, KW, _, stride, pad, dilation, groups = arg_lst[i]
        run(name, N, H, W, CO // CI, CI, KH, KW, stride, pad, dilation)

