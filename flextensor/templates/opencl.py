import math
from tvm import te
from functools import reduce
from . import utils
from flextensor.utils import is_power_of_x
from tvm.te import Stage
from tvm.tir import IterVar
from typing import Dict, List
from collections import defaultdict

__all__ = [
    'opencl_schedule',
    'opencl_schedule_bifrost',
]


def _lower_power2(x):
    p2 = 1
    while p2 <= x:
        p2 *= 2
    return p2 // 2


def _interleave_shift(l1, l2, p2):
    l1 = list(l1)
    l2 = list(l2)
    pub = list(zip(reversed(l1), reversed(l2[:p2])))
    n_pub = len(pub)
    pub = reversed([x for t in pub for x in t])
    pub = list(pub)
    if n_pub < p2:
        return l2[:p2-n_pub] + pub + l2[p2:]
    else:
        return l1[:len(l1) - n_pub] + pub + l2[p2:]


MAX_THREADS_PER_BLOCK = 384


class OpenCLScheduler:
    def __init__(self, config):
        self._ivs_extent: Dict[IterVar, int] = dict()
        self._thread_ivs: Dict[str, IterVar] = dict()
        self._config = defaultdict(lambda: None, config)

    def _reset(self, config):
        self._ivs_extent.clear()
        self._thread_ivs.clear()
        self._config = defaultdict(lambda: None, config)

    def _get_iv_extent(self, iv):
        if iv.dom is not None:
            return int(iv.dom.extent)
        return self._ivs_extent[iv]

    def _set_iv_extent(self, iv, ext):
        self._ivs_extent[iv] = ext

    def _split(self, stage: Stage, iv, factor=None, nparts=None):
        assert not (factor is None and nparts is None)

        iv_ext = self._get_iv_extent(iv)
        ivo, ivi = stage.split(iv, factor=factor, nparts=nparts)
        if factor is None:
            factor = (iv_ext + nparts - 1) // nparts
        elif nparts is None:
            nparts = (iv_ext + factor - 1) // factor
        self._set_iv_extent(ivo, nparts)
        self._set_iv_extent(ivi, factor)
        return ivo, ivi

    def _fuse(self, stage: Stage, *ivs):
        fused_ext = reduce(lambda a, b: a * b,
                           (self._get_iv_extent(iv) for iv in ivs), 1)
        fused_iv = stage.fuse(*ivs)
        self._set_iv_extent(fused_iv, fused_ext)
        return fused_iv

    def _bind(self, stage: Stage, iv, tiv):
        stage.bind(iv, tiv)
        self._thread_ivs[tiv.var.name] = iv

    def _split_axes(self, stage: Stage, axes, factors):
        if factors is None:
            return [[x] for x in axes]
        split_parts = []
        for iv, fs in zip(axes, factors):
            part = []
            for f in reversed(fs[1:]):
                iv, inner = self._split(stage, iv, factor=f)
                part.append(inner)
            part.append(iv)
            split_parts.append(list(reversed(part)))
        return split_parts

    def _fuse_axes(self, stage: Stage, iv_levels, fuse_pivots):
        if fuse_pivots is None:
            return iv_levels[:]
        fused_levels = []
        for level in iv_levels:
            beg = 0
            tmp_level = []
            for end in fuse_pivots:
                if end - beg > 1:
                    fused_iv = self._fuse(stage, *level[beg:end])
                    tmp_level.append(fused_iv)
                elif end - beg == 1:
                    tmp_level.append(level[beg])
                beg = end
            fused_levels.append(tmp_level)
        return fused_levels

    def _bind_axes(self, stage: Stage, iv_levels, tiv_levels):
        bind_tiv_levels = []
        n_level = len(iv_levels)
        if n_level == 1:
            bind_tiv_levels = [tiv_levels[0]]
        elif n_level == 2:
            bind_tiv_levels = [tiv_levels[0], tiv_levels[2]]
        else:
            bind_tiv_levels = tiv_levels[:]

        local_write_pos = None
        for iv_lv, tiv_lv in zip(iv_levels, bind_tiv_levels):
            assert len(iv_lv) <= len(tiv_lv)
            for iv, tiv in zip(iv_lv, tiv_lv):
                self._bind(stage, iv, tiv)
                local_write_pos = iv
        return local_write_pos

    def __call__(self, s: te.Schedule, op):
        write_cache = s.cache_write(op.output(0), "local")
        read_caches = [s.cache_read(t, "local", [write_cache]) for t in
                       op.input_tensors]

        op_stg: Stage = s[op]
        wc_stg: Stage = s[write_cache]
        rc_stgs: List[Stage] = [s[rc] for rc in read_caches]

        sp_ivs = [x for x in op_stg.op.axis]
        assert len(sp_ivs) > 0, "empty spatial axes"

        def set_pragma():
            n = sp_ivs[0]
            outer_scope, n = self._split(op_stg, n, nparts=1)
            sp_ivs[0] = n

            unroll = self._config["unroll"]
            if unroll is not None:
                step, explicit = unroll
                op_stg.pragma(outer_scope, 'auto_unroll_max_step', step)
                op_stg.pragma(outer_scope, 'unroll_explicit', explicit)
        set_pragma()

        def tile_and_fuse():
            sp_parts = self._split_axes(
                op_stg, sp_ivs, self._config["spatial"])
            sp_levels = list(zip(*sp_parts))
            op_stg.reorder(*(iv for lv in sp_levels for iv in lv))

            sp_levels = self._fuse_axes(
                op_stg, sp_levels, self._config["fuse"])
            return sp_levels
        sp_levels = tile_and_fuse()

        def bind_and_check():
            blocks = [te.thread_axis(f"blockIdx.{x}") for x in "xyz"]
            threads = [te.thread_axis(f"threadIdx.{x}") for x in "xyz"]
            vthreads = [te.thread_axis("vthread") for _ in "xyz"]
            local_write_pos = self._bind_axes(
                op_stg, sp_levels, [blocks, vthreads, threads])
            n_threads_per_block = reduce(lambda a, b: a * b, (self._get_iv_extent(
                iv) for (t, iv) in self._thread_ivs.items() if t.startswith("threadIdx")), 1)
            if n_threads_per_block > MAX_THREADS_PER_BLOCK:
                raise RuntimeError(
                    "Work group excess limit size: {} (required) vs. {} (given)".format(
                        n_threads_per_block, MAX_THREADS_PER_BLOCK))
            return local_write_pos
        local_write_pos = bind_and_check()

        def unroll_and_vectorize():
            bound_axes = set(self._thread_ivs.values())
            [op_stg.unroll(iv) for iv in sp_levels[-1]
             [:-1] if iv not in bound_axes]
            last_iv = sp_levels[-1][-1]
            if last_iv not in bound_axes:
                last_ext = self._get_iv_extent(last_iv)
                if last_ext > 16:
                    outer, inner = self._split(op_stg, last_iv, factor=16)
                    op_stg.unroll(outer)
                    op_stg.vectorize(inner)
                elif not is_power_of_x(2, last_ext):
                    outer, inner = self._split(op_stg, last_iv,
                                               factor=_lower_power2(last_ext))
                    op_stg.unroll(outer)
                    op_stg.vectorize(inner)
                else:
                    op_stg.vectorize(last_iv)
        unroll_and_vectorize()

        def handle_write_cache():
            # compute at
            wc_stg.compute_at(op_stg, local_write_pos)
            # split reduce axis
            re_ivs = wc_stg.op.reduce_axis
            re_parts = self._split_axes(
                wc_stg, re_ivs, self._config["reduce"])
            re_levels = list(zip(*re_parts))
            wc_sp_ivs = wc_stg.op.axis
            last_lv = re_levels[-1]
            # interleave reorder
            reorder_lst = [iv for lv in re_levels[:-1] for iv in lv]
            pos = self._config["reorder"]
            if pos is None:
                reorder_lst.extend(last_lv + wc_sp_ivs)
            else:
                reorder_lst.extend(_interleave_shift(last_lv, wc_sp_ivs, pos))
            wc_stg.reorder(*reorder_lst)
            # unroll
            [wc_stg.unroll(iv) for iv in wc_sp_ivs]

            return last_lv[-1]
        local_read_pos = handle_write_cache()

        def handle_read_caches():
            for rc_stg in rc_stgs:
                # compute at
                rc_stg.compute_at(wc_stg, local_read_pos)
                # unroll and vectorize
                rc_sp_ivs = rc_stg.op.axis
                [rc_stg.unroll(iv) for iv in rc_sp_ivs[:-1]]
                rc_stg.vectorize(rc_sp_ivs[-1])
        handle_read_caches()


def opencl_schedule_bifrost_v2(config, s, op, op_state):
    new_config = dict()
    if "spatial" in config and len(config["spatial"]) > 0:
        new_config["spatial"] = config["spatial"]
    if "reduce" in config and len(config["reduce"]) > 0:
        new_config["reduce"] = config["reduce"]
    if "unroll" in config and len(config["unroll"]) > 0:
        new_config["unroll"] = config["unroll"][0]
    if "fuse" in config and len(config["fuse"]) > 0:
        new_config["fuse"] = config["fuse"][0]
    if "reorder" in config and len(config["reorder"]) > 0:
        new_config["reorder"] = config["reorder"][0][0]
    OpenCLScheduler(new_config)(s, op)


def opencl_schedule_bifrost_v0(config, s, op, op_state):
    # always cache write here
    write_cache = s.cache_write(op.output(0), "local")

    # spatial split
    spatial_axes = [axis for axis in s[op].op.axis]
    assert len(spatial_axes) > 0, "empty spatial axes"  # must be non-empty
    n = spatial_axes[0]
    kernel_scope, n = s[op].split(n, nparts=1)
    spatial_axes[0] = n
    split_sp_axes, split_sp_exts = utils.split_axes(
        config, s, op, spatial_axes, "spatial")

    # always reorder here
    reorder_parts = list(zip(*split_sp_axes))
    reorder_part_exts = list(zip(*split_sp_exts))
    reorder_lst = reduce(lambda a, b: a + b, [list(p) for p in reorder_parts],
                         [])
    s[op].reorder(*reorder_lst)

    # handle fuse request and record op state
    fused_parts, fused_part_exts, fused_part_idx = utils.fuse_axes(
        config, s, op, op_state,
        reorder_parts, reorder_part_exts, reorder_lst)

    # always bind here
    # - prepare thread axis
    blocks = [te.thread_axis(f"blockIdx.{x}") for x in "xyz"]
    threads = [te.thread_axis(f"threadIdx.{x}") for x in "xyz"]
    vthreads = [te.thread_axis("vthread") for _ in "xyz"]
    block_exts = [-1, -1, -1]  # z, y, x
    vthread_exts = [-1, -1, -1]
    thread_exts = [-1, -1, -1]
    # - bind
    local_write_pos, bound_axes = utils.bind_axes(
        s, op, fused_parts, fused_part_exts,
        [blocks, vthreads, threads],
        [block_exts, vthread_exts, thread_exts])

    n_threads_per_block = reduce(
        lambda a, b: a * b, (ext for ext in thread_exts if ext != -1), 1)
    if n_threads_per_block > MAX_THREADS_PER_BLOCK:
        raise RuntimeError(
            "Work group excess limit size: {} (required) vs. {} (given)".format(
                n_threads_per_block, MAX_THREADS_PER_BLOCK))

    # unroll and vectorize
    [s[op].unroll(axis) for axis in fused_parts[-1][:-1] if
     axis not in bound_axes]
    last_part = fused_parts[-1][-1]
    if last_part not in bound_axes:
        last_ext = fused_part_exts[-1][-1]
        if last_ext > 16:
            outer, inner = s[op].split(last_part, factor=16)
            s[op].unroll(outer)
            s[op].vectorize(inner)
        elif not is_power_of_x(2, last_ext):
            outer, inner = s[op].split(last_part,
                                       factor=_lower_power2(last_ext))
            s[op].unroll(outer)
            s[op].vectorize(inner)
        else:
            s[op].vectorize(last_part)

    # always compute at here
    s[write_cache].compute_at(s[op], local_write_pos)

    # reduce_split
    reduce_axes = s[write_cache].op.reduce_axis
    split_re_axes, split_re_exts = utils.split_axes(
        config, s, write_cache, reduce_axes, "reduce")
    split_re_parts = list(zip(*split_re_axes))

    spatial_remainder = s[write_cache].op.axis

    # if has reduce axes
    if len(split_re_axes) > 0:
        # always reorder here
        reduce_reorder_parts = list(zip(*split_re_axes))
        last_part = reduce_reorder_parts[-1]
        reorder_lst = reduce(lambda a, b: a + b,
                             [list(p) for p in reduce_reorder_parts[:-1]], [])
        utils.interleave_reorder(config, s, write_cache,
                                 spatial_remainder, last_part, reorder_lst)

    # unroll
    [s[write_cache].unroll(sp) for sp in spatial_remainder]
    if len(split_re_axes) > 0:
        [s[write_cache].unroll(re) for re in split_re_axes[-1]]
    if "unroll" in config and len(config["unroll"]) > 0:
        step = config["unroll"][0][0]
        explicit = config["unroll"][0][1]
        s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        s[op].pragma(kernel_scope, 'unroll_explicit', explicit)


def opencl_schedule_bifrost_v1(config, s, op, op_state):
    # always cache write here
    write_cache = s.cache_write(op.output(0), "local")

    read_caches = [s.cache_read(t, "local", [write_cache]) for t in
                   op.input_tensors]

    # spatial split
    spatial_axes = [axis for axis in s[op].op.axis]
    assert len(spatial_axes) > 0, "empty spatial axes"  # must be non-empty
    n = spatial_axes[0]
    kernel_scope, n = s[op].split(n, nparts=1)
    spatial_axes[0] = n
    split_sp_axes, split_sp_exts = utils.split_axes(
        config, s, op, spatial_axes, "spatial")

    # always reorder here
    reorder_parts = list(zip(*split_sp_axes))
    reorder_part_exts = list(zip(*split_sp_exts))
    reorder_lst = reduce(lambda a, b: a + b, [list(p) for p in reorder_parts],
                         [])
    s[op].reorder(*reorder_lst)

    # handle fuse request and record op state
    fused_parts, fused_part_exts, fused_part_idx = utils.fuse_axes(
        config, s, op, op_state,
        reorder_parts, reorder_part_exts, reorder_lst)

    # always bind here
    # - prepare thread axis
    blocks = [te.thread_axis(f"blockIdx.{x}") for x in "xyz"]
    threads = [te.thread_axis(f"threadIdx.{x}") for x in "xyz"]
    vthreads = [te.thread_axis("vthread") for _ in "xyz"]
    block_exts = [-1, -1, -1]  # z, y, x
    vthread_exts = [-1, -1, -1]
    thread_exts = [-1, -1, -1]
    # - bind
    local_write_pos, bound_axes = utils.bind_axes(
        s, op, fused_parts, fused_part_exts,
        [blocks, vthreads, threads],
        [block_exts, vthread_exts, thread_exts])

    n_threads_per_block = reduce(
        lambda a, b: a * b, (ext for ext in thread_exts if ext != -1), 1)
    if n_threads_per_block > MAX_THREADS_PER_BLOCK:
        raise RuntimeError(
            "Work group excess limit size: {} (required) vs. {} (given)".format(
                n_threads_per_block, MAX_THREADS_PER_BLOCK))

    # print(block_exts, thread_exts, flush=True)

    # unroll and vectorize
    [s[op].unroll(axis) for axis in fused_parts[-1][:-1] if
     axis not in bound_axes]
    last_part = fused_parts[-1][-1]
    if last_part not in bound_axes:
        last_ext = fused_part_exts[-1][-1]
        if last_ext > 16:
            outer, inner = s[op].split(last_part, factor=16)
            s[op].unroll(outer)
            s[op].vectorize(inner)
        elif not is_power_of_x(2, last_ext):
            outer, inner = s[op].split(last_part,
                                       factor=_lower_power2(last_ext))
            s[op].unroll(outer)
            s[op].vectorize(inner)
        else:
            s[op].vectorize(last_part)

    # always compute at here
    s[write_cache].compute_at(s[op], local_write_pos)

    # reduce_split
    reduce_axes = s[write_cache].op.reduce_axis
    split_re_axes, split_re_exts = utils.split_axes(
        config, s, write_cache, reduce_axes, "reduce")
    split_re_parts = list(zip(*split_re_axes))

    spatial_remainder = s[write_cache].op.axis

    cache_pos = None
    # if has reduce axes
    if len(split_re_axes) > 0:
        n_re_level = len(split_re_axes[0])

        # always reorder here
        reduce_reorder_parts = list(zip(*split_re_axes))
        last_part = reduce_reorder_parts[-1]
        reorder_lst = reduce(lambda a, b: a + b,
                             [list(p) for p in reduce_reorder_parts[:-1]], [])
        utils.interleave_reorder(config, s, write_cache,
                                 spatial_remainder, last_part, reorder_lst)

        if n_re_level == 1:
            cache_pos = last_part[-1]
        else:
            mid = math.ceil(n_re_level / 2.0) - 1
            cache_pos = split_re_parts[mid][-1]

    if cache_pos is not None:
        for cache in read_caches:
            s[cache].compute_at(s[write_cache], cache_pos)
    else:
        for cache in read_caches:
            s[cache].compute_inline()

    # always cooperative fetching
    if cache_pos is not None:
        for cache in read_caches:
            fuse_lst = s[cache].op.axis
            fused = s[cache].fuse(*fuse_lst)
            count = 2
            cur = 1
            limit = 1024
            while count >= 0:
                factor = thread_exts[count]
                if factor < 0:
                    defined = False
                    factor = 16
                else:
                    defined = True
                cur *= factor
                if not defined and cur > limit:
                    break
                fused, inner = s[cache].split(fused, factor=factor)
                s[cache].bind(inner, threads[count])
                count -= 1

    # unroll
    [s[write_cache].unroll(sp) for sp in spatial_remainder]
    if len(split_re_axes) > 0:
        [s[write_cache].unroll(re) for re in split_re_axes[-1]]
    if "unroll" in config and len(config["unroll"]) > 0:
        step = config["unroll"][0][0]
        explicit = config["unroll"][0][1]
        s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        s[op].pragma(kernel_scope, 'unroll_explicit', explicit)


opencl_schedule_bifrost = opencl_schedule_bifrost_v2


def opencl_schedule(config, s, op, op_state):
    # assert_print(op in s)

    loop_lst = []
    loop_idx = []

    # always cache write here
    # if op.num_outputs > 1:
    #     raise RuntimeWarning("Too many outputs in one operation!")
    write_cache = s.cache_write(op.output(0), "local")
    # always cache read here
    read_cache_share_lst = []
    # read_cache_local_lst = []
    for t in op.input_tensors:
        share = s.cache_read(t, "shared", [write_cache])
        read_cache_share_lst.append(share)
        # local = s.cache_read(share, "local", [write_cache])
        # read_cache_local_lst.append(local)

    # spatial split
    spatial_axes = [axis for axis in s[op].op.axis]
    assert len(spatial_axes) > 0, "empty spatial axes"  # must be non-empty
    n = spatial_axes[0]
    kernel_scope, n = s[op].split(n, nparts=1)
    spatial_axes[0] = n
    splited_spatial_axes = []
    splited_spatial_extents = []
    if "spatial" in config and len(config["spatial"]) > 0:
        # to align each axis
        assert len(config["spatial"]) == len(spatial_axes), "align failed"
        for axis, nparts in zip(spatial_axes, config["spatial"]):
            axes, exts = [], []
            for count in range(len(nparts) - 1):
                outer, axis = s[op].split(axis, nparts=nparts[count])
                axes.append(outer)
                exts.append(nparts[count])
            axes.append(axis)
            exts.append(nparts[-1])
            splited_spatial_axes.append(axes)
            splited_spatial_extents.append(exts)
    else:
        for axis in spatial_axes:
            splited_spatial_axes.append([axis])
            splited_spatial_extents.append([axis.dom.extent.value])

    # always reorder here
    reorder_lst = []
    reorder_parts = []
    reorder_part_extents = []
    for count in range(len(splited_spatial_axes[0])):
        axes = [x[count] for x in splited_spatial_axes]
        exts = [x[count] for x in splited_spatial_extents]
        reorder_lst.extend(axes)
        reorder_parts.append(axes)
        reorder_part_extents.append(exts)
    s[op].reorder(*reorder_lst)
    # handle fuse request
    fused_parts = []
    fused_part_extents = []
    fused_part_idx = []
    if "fuse" in config and len(config["fuse"]) > 0:
        base_id = 0
        for part, extents in zip(reorder_parts, reorder_part_extents):
            tmp_part, exts, tmp_idx, idx, beg = [], [], [], 0, 0
            for end in config["fuse"][0]:
                if end - beg > 1:
                    fuse_lst = part[beg:end]
                    fused = s[op].fuse(*fuse_lst)
                    tmp_part.append(fused)
                    extent = reduce(lambda x, y: x * y, extents[beg:end], 1)
                    tmp_idx.extend([idx] * (end - beg))
                    idx += 1
                    exts.append(extent)
                elif end - beg == 1:
                    tmp_part.append(part[beg])
                    exts.append(extents[beg])
                    tmp_idx.append(idx)
                    idx += 1
                beg = end
            fused_parts.append(tmp_part)
            fused_part_extents.append(exts)
            fused_part_idx.append(tmp_idx)

            loop_lst.extend(tmp_part)
            loop_idx.extend([x + base_id for x in tmp_idx])
            base_id += len(tmp_part)
    else:
        fused_parts = reorder_parts
        fused_part_extents = reorder_part_extents
        fused_part_idx = [list(range(len(x))) for x in reorder_parts]

        loop_lst = reorder_lst
        loop_idx = list(range(len(reorder_lst)))
    # record op state
    op_state.loop_lst = loop_lst
    op_state.loop_idx = loop_idx

    # always bind here
    # - prepare thread axis
    blocks = [te.thread_axis(f"blockIdx.{x}") for x in "xyz"]
    threads = [te.thread_axis(f"threadIdx.{x}") for x in "xyz"]
    vthreads = [te.thread_axis("vthread") for _ in "xyz"]

    block_extents = [-1, -1, -1]  # z, y, x
    virtual_extents = [-1, -1, -1]
    thread_extents = [-1, -1, -1]

    bind_option = [None, None, None]
    bind_candidate = [blocks, vthreads, threads]
    candiate_extents = [block_extents, virtual_extents, thread_extents]

    # - bind
    num_parts = len(fused_parts)
    if num_parts == 1:
        bind_option[0] = (fused_parts[0], fused_part_extents[0])
        local_pos = fused_parts[0][:len(bind_candidate[0])][-1]
    elif num_parts == 2:
        bind_option[0] = (fused_parts[0], fused_part_extents[0])
        bind_option[2] = (fused_parts[1], fused_part_extents[1])
        local_pos = fused_parts[1][:len(bind_candidate[2])][-1]
    else:
        bind_option[0] = (fused_parts[0], fused_part_extents[0])
        bind_option[1] = (fused_parts[1], fused_part_extents[1])
        bind_option[2] = (fused_parts[2], fused_part_extents[2])
        local_pos = fused_parts[2][:len(bind_candidate[2])][-1]
    for option, candidate, extents in zip(bind_option, bind_candidate,
                                          candiate_extents):
        if option is not None:
            for i, axis in enumerate(option[0][:len(candidate)]):
                s[op].bind(axis, candidate[i])
                extents[i] = option[1][i]

    # compute at
    if "local_pos" in config and len(config["local_pos"]) > 0:
        local_at_part = config["local_pos"][0][0]
        local_at_idx = config["local_pos"][0][1]
        # index changed because of fusion
        cur_idx = fused_part_idx[local_at_part][local_at_idx]
        local_pos = fused_parts[local_at_part][cur_idx]

    # always compute at here
    s[write_cache].compute_at(s[op], local_pos)

    # reduce_split
    reduced_axes = s[write_cache].op.reduce_axis
    splited_reduced_axes = []
    if "reduce" in config and len(config["reduce"]) > 0:
        # to align each axis
        assert len(config["reduce"]) == len(
            reduced_axes), "align reduce failed"
        for axis, nparts in zip(reduced_axes, config["reduce"]):
            axes = []
            for count in range(len(nparts) - 1):
                outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                axes.append(outer)
            axes.append(axis)
            splited_reduced_axes.append(axes)
    else:
        for axis in reduced_axes:
            splited_reduced_axes.append([axis])
    share_pos = None
    # local_pos = None
    # if has reduce axes
    if len(splited_reduced_axes) > 0:
        # always reorder here
        reduced_nonfuse_lsts = []
        reorder_lst = []
        length = len(splited_reduced_axes[0])
        # leave the last part
        for count in range(length - 1):
            axes = [x[count] for x in splited_reduced_axes]
            reduced_nonfuse_lsts.append(axes)
            reorder_lst.extend(axes)
        # the last part
        last_part = [x[length - 1] for x in splited_reduced_axes]
        spatial_remainder = s[write_cache].op.axis
        # change the order of reduce axes and spatial axes
        if "reorder" in config and len(config["reorder"]) > 0:
            pos = config["reorder"][0][0]
            assert pos < len(spatial_remainder)
            axes = []
            count = len(spatial_remainder) - 1
            while count > pos:
                axes.append(spatial_remainder[count])
                count -= 1
            p = pos
            q = len(last_part) - 1
            while p >= 0 and q >= 0:
                axes.append(spatial_remainder[p])
                axes.append(last_part[q])
                p -= 1
                q -= 1
            while p >= 0:
                axes.append(spatial_remainder[p])
                p -= 1
            while q >= 0:
                axes.append(last_part[q])
                q -= 1
            axes = list(reversed(axes))
            reorder_lst.extend(axes)
        else:
            reorder_lst.extend(last_part)
            reorder_lst.extend(spatial_remainder)
        s[write_cache].reorder(*reorder_lst)
        # decide where to compute at
        if "share_pos" in config and len(config["share_pos"]) > 0:
            share_at = config["share_pos"][0][0]
            share_idx = config["share_pos"][0][1]
            reduced_nonfuse_lsts.append(last_part)
            share_pos = reduced_nonfuse_lsts[share_at][share_idx]
        else:
            if length == 1:
                share_pos = last_part[-1]
            else:
                mid = math.ceil(length / 2.0) - 1
                share_pos = reduced_nonfuse_lsts[mid][-1]
                # local_pos = last_part[-1]

    # always cache read here
    if share_pos is not None:
        for share in read_cache_share_lst:
            s[share].compute_at(s[write_cache], share_pos)
    else:
        for share in read_cache_share_lst:
            s[share].compute_inline()
    # if local_pos is not None:
    #     for local in read_cache_local_lst:
    #         s[local].compute_at(s[write_cache], local_pos)
    # else:
    #     for local in read_cache_local_lst:
    #         s[local].compute_inline()

    # always cooperative fetching
    if share_pos is not None:
        for share in read_cache_share_lst:
            fuse_lst = s[share].op.axis
            fused = s[share].fuse(*fuse_lst)
            count = 2
            cur = 1
            limit = 1024
            while count >= 0:
                factor = thread_extents[count]
                if factor < 0:
                    defined = False
                    factor = 16
                else:
                    defined = True
                cur *= factor
                if not defined and cur > limit:
                    break
                fused, inner = s[share].split(fused, factor=factor)
                s[share].bind(inner, threads[count])
                count -= 1

    # unroll
    if "unroll" in config and len(config["unroll"]) > 0:
        step = config["unroll"][0][0]
        explicit = config["unroll"][0][1]
        s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        s[op].pragma(kernel_scope, 'unroll_explicit', explicit)
