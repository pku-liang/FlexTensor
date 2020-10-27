import math
from tvm import te
from functools import reduce
from . import utils

__all__ = [
    'opencl_schedule',
    'opencl_schedule_bifrost',
]


def opencl_schedule_bifrost(config, s, op, op_state):
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
    reorder_lst = reduce(lambda a, b: a + b, [list(p) for p in reorder_parts], [])
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

    # unroll and vectorize
    [s[op].unroll(axis) for axis in fused_parts[-1][:-1] if axis not in bound_axes]
    if fused_parts[-1][-1] not in bound_axes:
        s[op].vectorize(fused_parts[-1][-1])

    # always compute at here
    s[write_cache].compute_at(s[op], local_write_pos)

    # reduce_split
    reduce_axes = s[write_cache].op.reduce_axis
    split_re_axes, split_re_exts = utils.split_axes(
        config, s, write_cache, reduce_axes, "reduce")

    spatial_remainder = s[write_cache].op.axis
    # if has reduce axes
    if len(split_re_axes) > 0:
        # always reorder here
        reduce_reorder_parts = list(zip(*split_re_axes))
        last_part = reduce_reorder_parts[-1]
        reorder_lst = reduce(lambda a, b: a + b, [list(p) for p in reduce_reorder_parts[:-1]], [])
        utils.interleave_reorder(config, s, write_cache,
                                 spatial_remainder, last_part, reorder_lst)

    # unroll
    [s[write_cache].unroll(sp) for sp in spatial_remainder]
    [s[write_cache].unroll(re) for re in split_re_axes[-1]]
    if "unroll" in config and len(config["unroll"]) > 0:
        step = config["unroll"][0][0]
        explicit = config["unroll"][0][1]
        s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        s[op].pragma(kernel_scope, 'unroll_explicit', explicit)


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
    for option, candidate, extents in zip(bind_option, bind_candidate, candiate_extents):
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
        assert len(config["reduce"]) == len(reduced_axes), "align reduce failed"
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
