import os
import tvm
import topi
import numpy as np
from tvm import rpc
from topi.util import simplify
from topi.nn.util import get_pad_tuple


def to_int(expr):
    res = int(expr)
    return res


def to_tuple(expr_tuple):
    return tuple([to_int(x) for x in expr_tuple])


def conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = topi.nn.pad(Input, pad_before, pad_after, name="pad_temp")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    return tvm.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: tvm.sum(
            temp[nn, rc, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="conv2d_nchw")


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


def schedule_conv2d_hwcn(outs, args_lst=None):
    """Schedule for conv2d_hwcn and any element-wise operations.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_hwcn in the format
        of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_hwcn.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    sch = tvm.create_schedule([x.op for x in outs])
    def schedule(Apad, W, B):
        """Schedule conv2d_hwcn"""
        sch[Apad].compute_inline()
        AA = sch.cache_read(Apad, "shared", [B])
        WW = sch.cache_read(W, "shared", [B])
        AL = sch.cache_read(AA, "local", [B])
        WL = sch.cache_read(WW, "local", [B])

        if B.op in sch.outputs:
            Out = B
            BL = sch.cache_write(Out, "local")
        else:
            Out = sch.outputs[0].output(0)
            sch[B].set_scope("local")
            BL = B

        if args_lst is not None:
            tile, num_thread, step, vthread = args_lst
        else:
            tile = 8
            num_thread = 8
            step = 8
            vthread = 2
        block_factor = tile * num_thread

        block_x = tvm.thread_axis("blockIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        block_z = tvm.thread_axis("blockIdx.z")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
        thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
        thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

        hi, wi, fi, ni = sch[Out].op.axis
        bz = sch[Out].fuse(hi, wi)
        by, fi = sch[Out].split(fi, factor=block_factor)
        bx, ni = sch[Out].split(ni, factor=block_factor)
        tyz, fi = sch[Out].split(fi, nparts=vthread)
        txz, ni = sch[Out].split(ni, nparts=vthread)
        ty, fi = sch[Out].split(fi, nparts=num_thread)
        tx, ni = sch[Out].split(ni, nparts=num_thread)
        sch[Out].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)
        sch[Out].bind(bz, block_z)
        sch[Out].bind(by, block_y)
        sch[Out].bind(bx, block_x)
        sch[Out].bind(tyz, thread_yz)
        sch[Out].bind(txz, thread_xz)
        sch[Out].bind(ty, thread_y)
        sch[Out].bind(tx, thread_x)

        # Schedule BL local write
        sch[BL].compute_at(sch[Out], tx)
        yi, xi, fi, ni = sch[BL].op.axis
        ry, rx, rc = sch[BL].op.reduce_axis
        rco, rci = sch[BL].split(rc, factor=step)
        sch[BL].reorder(rco, ry, rx, rci, fi, ni)
        fuse_index = sch[BL].fuse(ry, rx)
        fuse_index = sch[BL].fuse(fuse_index, rco)
        rx = fuse_index

        sch[AA].compute_at(sch[BL], rx)
        sch[WW].compute_at(sch[BL], rx)
        sch[AL].compute_at(sch[BL], rci)
        sch[WL].compute_at(sch[BL], rci)
        # Schedule for A's shared memory load
        yi, xi, ci, ni = sch[AA].op.axis
        ty, ci = sch[AA].split(ci, nparts=num_thread)
        tx, ni = sch[AA].split(ni, nparts=num_thread)
        _, ni = sch[AA].split(ni, factor=4)
        sch[AA].reorder(ty, tx, yi, xi, ci, ni)
        sch[AA].bind(ty, thread_y)
        sch[AA].bind(tx, thread_x)
        sch[AA].vectorize(ni)
        # Schedule for W's shared memory load
        yi, xi, ci, fi = sch[WW].op.axis
        ty, ci = sch[WW].split(ci, nparts=num_thread)
        tx, fi = sch[WW].split(fi, nparts=num_thread)
        _, fi = sch[WW].split(fi, factor=4)
        sch[WW].reorder(ty, tx, yi, xi, ci, fi)
        sch[WW].bind(ty, thread_y)
        sch[WW].bind(tx, thread_x)
        sch[WW].vectorize(fi)

    scheduled_ops = []

    def traverse(operator):
        """Traverse operators from computation graph"""
        if topi.tag.is_broadcast(operator.tag):
            if operator not in sch.outputs:
                sch[operator].compute_inline()
            for tensor in operator.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        elif operator.tag == 'conv2d_hwcn':
            Apad = operator.input_tensors[0]
            W = operator.input_tensors[1]
            if isinstance(W.op, tvm.tensor.ComputeOp) and 'dilate' in W.op.tag:
                sch[W].compute_inline()
            B = operator.output(0)
            schedule(Apad, W, B)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

        scheduled_ops.append(operator)

    traverse(outs[0].op)
    return sch


yolo_shapes = [
    # yolo
    (1, 3, 448, 448, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 192, 56, 56, 128, 192, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1),  # conv4   3
    (1, 256, 56, 56, 256, 256, 1, 1, 1, 1, 0, 1, 1),  # conv5   4
    (1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv7   6
    (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv8   7
    # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv9
    # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv10
    # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv11
    # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv12
    # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv13
    # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv14
    (1, 512, 28, 28, 512, 512, 1, 1, 1, 1, 0, 1, 1),  # conv15      8
    (1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv16     9
    (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv17    10
    (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv18     11
    # (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv19
    # (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv20
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv21   12
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1),  # conv22   13
    (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv23     14
    # (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv24
]


LOCAL_RPC = False
LIB_DIR = "."


def evaluate(name, s, bufs, target, dev_id, number, rpc_info=None):
    if rpc_info is not None:
        host = rpc_info.host
        port = rpc_info.port
    else:
        # local
        host = "0.0.0.0"
        port = 9090     # default port
    if host == "0.0.0.0":
        if LOCAL_RPC:
            use_rpc = True
        else:
            use_rpc = False
    else:
        use_rpc = True
    if use_rpc:
        remote = rpc.connect(host, port)
        ctx = remote.context(target, dev_id)
    else:
        ctx = tvm.context(target, dev_id)
    tvm_arys = []
    for buf in bufs:
        shape = to_tuple(buf.shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(buf.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    try:
        func_file = "{}.tar".format(name)
        if rpc_info is not None and rpc_info.target_host is not None:
            func = tvm.build(s, bufs, target=target, target_host=rpc_info.target_host)
        else:
            func = tvm.build(s, bufs, target=target)
        if use_rpc:
            func.export_library(os.path.join(LIB_DIR, func_file))
            remote.upload(os.path.join(LIB_DIR, func_file))
            func = remote.load_module(func_file)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
        time_cost = evaluator(*tvm_arys).mean * 1e3
    except Exception as e:
        # print(e)
        print("Bad schedule")
        time_cost = float("inf")
    finally:
        while len(tvm_arys) > 0:
            del tvm_arys[-1]
        if os.path.exists(os.path.join(LIB_DIR, func_file)):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
        elif os.path.exists(os.path.join(LIB_DIR, func_file + ".so")):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
    return time_cost


if __name__ == "__main__":
    shapes = yolo_shapes
    names = ["yolo_" + str(i) for i in range(len(shapes))]
    for name, shape in zip(names, shapes):
        print("%s begins..." % name)
        N, C, H, W, K, _, k_h, k_w, _, st, pad, dilation, groups = shape
        N = 64
        Img = tvm.placeholder([H, W, C, N])
        Filter = tvm.placeholder([k_h, k_w, C, K])
        Output = conv2d_hwcn(Img, Filter, st, pad, dilation, 'float32')
        bufs = [Img, Filter, Output]

        best_cost = float("inf")
        best_params = None
        for tile in [8, 16, 32]:
            for num_thread in [8, 16, 32]:
                for step in [8, 16, 32]:
                    for vthread in [2, 4]:
                        arg_lst = [tile, num_thread, step, vthread]
                        s = schedule_conv2d_hwcn(Output, arg_lst)
                        cost = evaluate(name, s, bufs, "cuda", 0, 10, None)
                        if cost < best_cost:
                            best_cost = cost
                            best_params = arg_lst.copy()
                        print("    %s use %fms" % (name, cost))
        print("%s best use %fms [%d, %d, %d, %d]" % (name, best_cost, *best_params))
        