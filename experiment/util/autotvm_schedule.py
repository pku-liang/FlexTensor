import tvm
from tvm import autotvm
import topi
import topi.testing as testing
import auto_schedule.task as task
import numpy as np
from experiment.util import autotvm_ops


@autotvm.template
def conv2d_cuda_schedule(N, C, H, W, K, Hk, stride, padding, dilation, groups):
    ops, [data, kernel, conv] = task.conv2d(
        N, C, H, W, K, Hk, stride, padding, dilation, groups)
    s = tvm.create_schedule(ops)
    cfg = autotvm.get_config()

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis

    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # inline padding
    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, 'local')

    # create cache stage
    AA = s.cache_read(data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])
    AL = s.cache_read(AA, 'local', [OL])
    WL = s.cache_read(WW, 'local', [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
    kernel_scope = n  # this is the scope to attach global config inside this kernel

    s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
    s[output].bind(by, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
    ryo, rym, ryi = cfg['tile_rx'].apply(s, OL, ry)
    rxo, rxm, rxi = cfg['tile_ry'].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

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

    # tune unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                     cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit',
                     cfg['unroll_explicit'].val)

    return s, [raw_data, kernel, conv]


def conv2d_evaluate(func, shape, target, dev):
    N, C, H, W, K, Hk, stride, padding, dilation, groups = shape
    Ho = (H + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1
    Wo = (W + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1

    a_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(K, C // groups, Hk, Hk)).astype(np.float32)
    c_shape = (N, K, Ho, Wo)

    if target == 'cuda':
        ctx = tvm.gpu(dev)
    else:
        ctx = tvm.cpu(dev)
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_shape, ctx=ctx)
    func(a_tvm, w_tvm, c_tvm)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    return evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3


conv2d_schedule_dispatcher = {
    "cuda": conv2d_cuda_schedule,
    "llvm": None,
}


@autotvm.template
def conv_transpose2d_cuda_schedule(N, C, H, W, K, Hk, stride, padding, dilation):
    data = tvm.placeholder((N, C, H, W))
    kernel = tvm.placeholder((C, K, Hk, Hk))
    cfg = autotvm.get_config()
    conv = autotvm_ops.conv_transpose2d(
        cfg, data, kernel, (stride,) * 2, (padding,) * 2)
    ops = [conv.op]
    s = tvm.create_schedule(ops)

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis

    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # inline padding
    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, 'local')

    # create cache stage
    AA = s.cache_read(data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])
    AL = s.cache_read(AA, 'local', [OL])
    WL = s.cache_read(WW, 'local', [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
    kernel_scope = n  # this is the scope to attach global config inside this kernel

    s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
    s[output].bind(by, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
    ryo, rym, ryi = cfg['tile_rx'].apply(s, OL, ry)
    rxo, rxm, rxi = cfg['tile_ry'].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

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

    # tune unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                     cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit',
                     cfg['unroll_explicit'].val)

    return s, [raw_data, kernel, conv]


def conv_transpose2d_evaluate(func, shape, target, dev):
    N, C, H, W, K, Hk, stride, padding, dilation = shape
    Ho = (H - 1) * stride - 2 * padding + dilation * (Hk - 1) + 1
    Wo = (W - 1) * stride - 2 * padding + dilation * (Hk - 1) + 1

    a_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(C, K, Hk, Hk)).astype(np.float32)
    c_shape = (N, K, Ho, Wo)

    if target == 'cuda':
        ctx = tvm.gpu(dev)
    else:
        ctx = tvm.cpu(dev)
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_shape, ctx=ctx)
    func(a_tvm, w_tvm, c_tvm)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    return evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3


conv_transpose2d_schedule_dispatcher = {
    'cuda': conv_transpose2d_cuda_schedule,
    'llvm': None,
}


@autotvm.template
def conv1d_cuda_schedule(N, C, L, K, Lk, stride, padding, dilation):
    ops, [data, kernel, conv] = task.conv1d(
        N, C, L, K, Lk, stride, padding, dilation)
    s = tvm.create_schedule(ops)
    cfg = autotvm.get_config()

    n, c, l = s[conv].op.axis
    rc, rl = s[conv].op.reduce_axis

    cfg.define_split("tile_c", c, num_outputs=4)
    cfg.define_split("tile_l", l, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_rl", rl, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])

    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, "local")

    AA = s.cache_read(data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])
    AL = s.cache_read(AA, "local", [OL])
    WL = s.cache_read(WW, "local", [OL])

    n, c, l = s[output].op.axis
    bc, vc, tc, ci = cfg['tile_c'].apply(s, output, c)
    bl, vl, tl, li = cfg['tile_l'].apply(s, output, l)
    kernel_scope = n

    s[output].bind(bc, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bl, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vc, tvm.thread_axis("vthread"))
    s[output].bind(vl, tvm.thread_axis("vthread"))
    s[output].bind(tc, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tl, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(n, bc, bl, vc, vl, tc, tl, ci, li)
    s[OL].compute_at(s[output], tl)

    n, c, l = s[OL].op.axis
    rc, rl = s[OL].op.reduce_axis
    rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
    rlo, rlm, rli = cfg["tile_rl"].apply(s, OL, rl)
    s[OL].reorder(rco, rlo, rcm, rlm, rci, rli, n, c, l)

    s[AA].compute_at(s[OL], rlo)
    s[WW].compute_at(s[OL], rlo)
    s[AL].compute_at(s[OL], rlm)
    s[WL].compute_at(s[OL], rlm)

    for load in [AA, WW]:
        n, c, l = s[load].op.axis
        fused = s[load].fuse(n, c, l)
        tc, fused = s[load].split(fused, nparts=cfg['tile_c'].size[2])
        tl, fused = s[load].split(fused, nparts=cfg['tile_l'].size[2])
        s[load].bind(tc, tvm.thread_axis('threadIdx.y'))
        s[load].bind(tl, tvm.thread_axis('threadIdx.x'))

    s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                     cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit',
                     cfg['unroll_explicit'].val)

    return s, [raw_data, kernel, conv]


def conv1d_evaluate(func, shape, target, dev):
    N, C, L, K, Lk, stride, padding, dilation = shape
    Lo = (L + 2 * padding - dilation * (Lk - 1) - 1) // stride + 1

    a_np = np.random.uniform(size=(N, C, L)).astype(np.float32)
    w_np = np.random.uniform(size=(K, C, Lk)).astype(np.float32)
    c_shape = (N, K, Lo)

    if target == 'cuda':
        ctx = tvm.gpu(dev)
    else:
        ctx = tvm.cpu(dev)

    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_shape, ctx=ctx)
    func(a_tvm, w_tvm, c_tvm)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    return evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3


conv1d_schedule_dispatcher = {
    'cuda': conv1d_cuda_schedule,
    'llvm': None
}


@autotvm.template
def conv_transpose1d_cuda_schedule(N, C, L, K, Lk, stride, padding, dilation):
    data = tvm.placeholder((N, C, L))
    kernel = tvm.placeholder((C, K, Lk))
    cfg = autotvm.get_config()
    conv = autotvm_ops.conv_transpose1d(
        cfg, data, kernel, (stride,), (padding,))
    ops = [conv.op]
    s = tvm.create_schedule(ops)

    n, c, l = s[conv].op.axis
    rc, rl = s[conv].op.reduce_axis

    cfg.define_split("tile_c", c, num_outputs=4)
    cfg.define_split("tile_l", l, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_rl", rl, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])

    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, "local")

    AA = s.cache_read(data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])
    AL = s.cache_read(AA, "local", [OL])
    WL = s.cache_read(WW, "local", [OL])

    n, c, l = s[output].op.axis
    bc, vc, tc, ci = cfg['tile_c'].apply(s, output, c)
    bl, vl, tl, li = cfg['tile_l'].apply(s, output, l)
    kernel_scope = n

    s[output].bind(bc, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bl, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vc, tvm.thread_axis("vthread"))
    s[output].bind(vl, tvm.thread_axis("vthread"))
    s[output].bind(tc, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tl, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(n, bc, bl, vc, vl, tc, tl, ci, li)
    s[OL].compute_at(s[output], tl)

    n, c, l = s[OL].op.axis
    rc, rl = s[OL].op.reduce_axis
    rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
    rlo, rlm, rli = cfg["tile_rl"].apply(s, OL, rl)
    s[OL].reorder(rco, rlo, rcm, rlm, rci, rli, n, c, l)

    s[AA].compute_at(s[OL], rlo)
    s[WW].compute_at(s[OL], rlo)
    s[AL].compute_at(s[OL], rlm)
    s[WL].compute_at(s[OL], rlm)

    for load in [AA, WW]:
        n, c, l = s[load].op.axis
        fused = s[load].fuse(n, c, l)
        tc, fused = s[load].split(fused, nparts=cfg['tile_c'].size[2])
        tl, fused = s[load].split(fused, nparts=cfg['tile_l'].size[2])
        s[load].bind(tc, tvm.thread_axis('threadIdx.y'))
        s[load].bind(tl, tvm.thread_axis('threadIdx.x'))

    s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                     cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit',
                     cfg['unroll_explicit'].val)

    return s, [raw_data, kernel, conv]


def conv_transpose1d_evaluate(func, shape, target, dev):
    N, C, L, K, Lk, stride, padding, dilation = shape
    Lo = (L - 1) * stride - 2 * padding + dilation * (Lk - 1) + 1

    a_np = np.random.uniform(size=(N, C, L)).astype(np.float32)
    w_np = np.random.uniform(size=(C, K, Lk)).astype(np.float32)
    c_shape = (N, K, Lo)

    if target == 'cuda':
        ctx = tvm.gpu(dev)
    else:
        ctx = tvm.cpu(dev)

    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_shape, ctx=ctx)
    func(a_tvm, w_tvm, c_tvm)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    return evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3


conv_transpose1d_schedule_dispatcher = {
    "cuda": conv_transpose1d_cuda_schedule,
    "llvm": None
}


@autotvm.template
def conv3d_cuda_schedule(N, C, D, H, W, K, Hk, stride, padding, dilation):
    ops, [data, kernel, conv] = task.conv3d(
        N, C, D, H, W, K, Hk, stride, padding, dilation)
    s = tvm.create_schedule(ops)
    cfg = autotvm.get_config()

    n, c, d, h, w = s[conv].op.axis
    rc, rd, rh, rw = s[conv].op.reduce_axis

    cfg.define_split("tile_c", c, num_outputs=4)
    cfg.define_split("tile_d", d, num_outputs=4)
    cfg.define_split("tile_h", h, num_outputs=4)
    cfg.define_split("tile_w", w, num_outputs=4)
    cfg.define_split('tile_rc', rc, num_outputs=3)
    cfg.define_split('tile_rd', rd, num_outputs=3)
    cfg.define_split('tile_rh', rh, num_outputs=3)
    cfg.define_split('tile_rw', rw, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])

    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, 'local')

    AA = s.cache_read(data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])
    AL = s.cache_read(AA, 'local', [OL])
    WL = s.cache_read(WW, 'local', [OL])

    n, c, d, h, w = s[output].op.axis
    bc, vc, tc, ci = cfg['tile_c'].apply(s, output, c)
    bd, vd, td, di = cfg['tile_d'].apply(s, output, d)
    bh, vh, th, hi = cfg['tile_h'].apply(s, output, h)
    bw, vw, tw, wi = cfg['tile_w'].apply(s, output, w)
    kernel_scope = n

    s[output].bind(bc, tvm.thread_axis("blockIdx.z"))
    s[output].bind(bd, tvm.thread_axis("blockIdx.z"))
    s[output].bind(bh, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bw, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vc, tvm.thread_axis("vthread"))
    s[output].bind(vd, tvm.thread_axis("vthread"))
    s[output].bind(vh, tvm.thread_axis("vthread"))
    s[output].bind(vw, tvm.thread_axis("vthread"))
    s[output].bind(tc, tvm.thread_axis("threadIdx.z"))
    s[output].bind(td, tvm.thread_axis("threadIdx.z"))
    s[output].bind(th, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tw, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(n, bc, bd, bh, bw, vc, vd, vh, vw,
                      tc, td, th, tw, ci, di, hi, wi)
    s[OL].compute_at(s[output], tw)

    n, c, d, h, w = s[OL].op.axis
    rc, rd, rh, rw = s[OL].op.reduce_axis
    rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
    rdo, rdm, rdi = cfg['tile_rd'].apply(s, OL, rd)
    rho, rhm, rhi = cfg['tile_rh'].apply(s, OL, rh)
    rwo, rwm, rwi = cfg['tile_rw'].apply(s, OL, rw)
    s[OL].reorder(rco, rdo, rho, rwo, rcm, rdm, rhm, rwm, rci, rdi, rhi, rwi)

    s[AA].compute_at(s[OL], rwo)
    s[WW].compute_at(s[OL], rwo)
    s[AL].compute_at(s[OL], rwm)
    s[WL].compute_at(s[OL], rwm)

    for load in [AA, WW]:
        n, c, d, h, w = s[load].op.axis
        fused = s[load].fuse(n, c, d, h, w)
        tc, fused = s[load].split(fused, nparts=cfg['tile_c'].size[2])
        td, fused = s[load].split(fused, nparts=cfg['tile_d'].size[2])
        th, fused = s[load].split(fused, nparts=cfg['tile_h'].size[2])
        tw, fused = s[load].split(fused, nparts=cfg['tile_w'].size[2])
        s[load].bind(tc, tvm.thread_axis("threadIdx.z"))
        s[load].bind(td, tvm.thread_axis("threadIdx.z"))
        s[load].bind(th, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tw, tvm.thread_axis("threadIdx.x"))

    s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                     cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit',
                     cfg['unroll_explicit'].val)

    return s, [raw_data, kernel, conv]


def conv3d_evaluate(func, shape, target, dev):
    N, C, D, H, W, K, Hk, stride, padding, dilation = shape
    Do = (D + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1
    Ho = (H + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1
    Wo = (W + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1

    a_np = np.random.uniform(size=(N, C, D, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(K, C, Hk, Hk, Hk)).astype(np.float32)
    c_shape = (N, K, Do, Ho, Wo)

    if target == 'cuda':
        ctx = tvm.gpu(dev)
    else:
        ctx = tvm.cpu(dev)

    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_shape, ctx=ctx)
    func(a_tvm, w_tvm, c_tvm)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    return evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3


conv3d_schedule_dispatcher = {
    "cuda": conv3d_cuda_schedule,
    "llvm": None
}


@autotvm.template
def conv_transpose3d_cuda_schedule(N, C, D, H, W, K, Hk, stride, padding, dilation):
    data = tvm.placeholder((N, C, D, H, W))
    kernel = tvm.placeholder((C, K, Hk, Hk, Hk))
    cfg = autotvm.get_config()
    conv = autotvm_ops.conv_transpose3d(
        cfg, data, kernel, (stride,) * 3, (padding,) * 3)
    ops = [conv.op]
    s = tvm.create_schedule(ops)

    n, c, d, h, w = s[conv].op.axis
    rc, rd, rh, rw = s[conv].op.reduce_axis

    cfg.define_split("tile_c", c, num_outputs=4)
    cfg.define_split("tile_d", d, num_outputs=4)
    cfg.define_split("tile_h", h, num_outputs=4)
    cfg.define_split("tile_w", w, num_outputs=4)
    cfg.define_split('tile_rc', rc, num_outputs=3)
    cfg.define_split('tile_rd', rd, num_outputs=3)
    cfg.define_split('tile_rh', rh, num_outputs=3)
    cfg.define_split('tile_rw', rw, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])

    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, 'local')

    AA = s.cache_read(data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])
    AL = s.cache_read(AA, 'local', [OL])
    WL = s.cache_read(WW, 'local', [OL])

    n, c, d, h, w = s[output].op.axis
    bc, vc, tc, ci = cfg['tile_c'].apply(s, output, c)
    bd, vd, td, di = cfg['tile_d'].apply(s, output, d)
    bh, vh, th, hi = cfg['tile_h'].apply(s, output, h)
    bw, vw, tw, wi = cfg['tile_w'].apply(s, output, w)
    kernel_scope = n

    s[output].bind(bc, tvm.thread_axis("blockIdx.z"))
    s[output].bind(bd, tvm.thread_axis("blockIdx.z"))
    s[output].bind(bh, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bw, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vc, tvm.thread_axis("vthread"))
    s[output].bind(vd, tvm.thread_axis("vthread"))
    s[output].bind(vh, tvm.thread_axis("vthread"))
    s[output].bind(vw, tvm.thread_axis("vthread"))
    s[output].bind(tc, tvm.thread_axis("threadIdx.z"))
    s[output].bind(td, tvm.thread_axis("threadIdx.z"))
    s[output].bind(th, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tw, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(n, bc, bd, bh, bw, vc, vd, vh, vw,
                      tc, td, th, tw, ci, di, hi, wi)
    s[OL].compute_at(s[output], tw)

    n, c, d, h, w = s[OL].op.axis
    rc, rd, rh, rw = s[OL].op.reduce_axis
    rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
    rdo, rdm, rdi = cfg['tile_rd'].apply(s, OL, rd)
    rho, rhm, rhi = cfg['tile_rh'].apply(s, OL, rh)
    rwo, rwm, rwi = cfg['tile_rw'].apply(s, OL, rw)
    s[OL].reorder(rco, rdo, rho, rwo, rcm, rdm, rhm, rwm, rci, rdi, rhi, rwi)

    s[AA].compute_at(s[OL], rwo)
    s[WW].compute_at(s[OL], rwo)
    s[AL].compute_at(s[OL], rwm)
    s[WL].compute_at(s[OL], rwm)

    for load in [AA, WW]:
        n, c, d, h, w = s[load].op.axis
        fused = s[load].fuse(n, c, d, h, w)
        tc, fused = s[load].split(fused, nparts=cfg['tile_c'].size[2])
        td, fused = s[load].split(fused, nparts=cfg['tile_d'].size[2])
        th, fused = s[load].split(fused, nparts=cfg['tile_h'].size[2])
        tw, fused = s[load].split(fused, nparts=cfg['tile_w'].size[2])
        s[load].bind(tc, tvm.thread_axis("threadIdx.z"))
        s[load].bind(td, tvm.thread_axis("threadIdx.z"))
        s[load].bind(th, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tw, tvm.thread_axis("threadIdx.x"))

    s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                     cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit',
                     cfg['unroll_explicit'].val)

    return s, [raw_data, kernel, conv]


def conv_transpose3d_evaluate(func, shape, target, dev):
    N, C, D, H, W, K, Hk, stride, padding, dilation = shape
    Do = (D - 1) * stride - 2 * padding + dilation * (Hk - 1) + 1
    Ho = (H - 1) * stride - 2 * padding + dilation * (Hk - 1) + 1
    Wo = (W - 1) * stride - 2 * padding + dilation * (Hk - 1) + 1

    a_np = np.random.uniform(size=(N, C, D, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(C, K, Hk, Hk, Hk)).astype(np.float32)
    c_shape = (N, K, Do, Ho, Wo)

    if target == 'cuda':
        ctx = tvm.gpu(dev)
    else:
        ctx = tvm.cpu(dev)

    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_shape, ctx=ctx)
    func(a_tvm, w_tvm, c_tvm)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    return evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3


conv_transpose3d_schedule_dispatcher = {
    'cuda': conv_transpose3d_cuda_schedule,
    'llvm': None,
}
