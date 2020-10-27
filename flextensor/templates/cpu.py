from flextensor.utils import assert_print
from functools import reduce

__all__ = [
    'cpu_schedule_split_fuse',
    'cpu_schedule_split_reorder_fuse',
    'cpu_schedule_simple',
]


def cpu_schedule_split_fuse(config, s, op, op_state):
    # always cache write here
    # if op.num_outputs > 1:
    #     raise RuntimeWarning("Too many outputs in one operation!")
    write_cache = s.cache_write(op.output(0), "global")

    # spatial split
    spatial_axes = s[op].op.axis
    splited_spatial_axes = []
    if "spatial" in config and len(config["spatial"]) > 0:
        # to align each axis
        assert_print(len(config["spatial"]) == len(spatial_axes), "align failed")
        for axis, nparts in zip(spatial_axes, config["spatial"]):
            tmp_buffer = []
            for count in range(len(nparts) - 1):
                outer, axis = s[op].split(axis, nparts=nparts[count])
                tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_spatial_axes.append(tmp_buffer)
    else:
        for axis in spatial_axes:
            splited_spatial_axes.append([axis])
    assert_print(len(splited_spatial_axes) > 0, "empty spatial axes")  # must be non-empty

    # always reorder and fuse here
    spatial_fuse_lsts = []
    spatial_fuse_extents = []
    reorder_lst = []
    fused_spatial_axes = []
    for count in range(len(splited_spatial_axes[0])):
        tmp_buffer = [x[count] for x in splited_spatial_axes]
        tmp_extent = reduce(lambda a, b: a * b, [x[count] for x in config["spatial"]])
        spatial_fuse_lsts.append(tmp_buffer)
        spatial_fuse_extents.append(tmp_extent)
        reorder_lst.extend(tmp_buffer)
    s[op].reorder(*reorder_lst)
    for fuse_lst in spatial_fuse_lsts:
        fused = s[op].fuse(*fuse_lst)
        fused_spatial_axes.append(fused)
    kernel_scope = fused_spatial_axes[0]

    # always parallel here
    length = len(fused_spatial_axes)
    assert_print(length > 0, "empty spatial axes!")
    s[op].parallel(fused_spatial_axes[0])
    if length == 1:
        thread_pos = fused_spatial_axes[0]
    if 2 <= length <= 3:
        thread_pos = fused_spatial_axes[1]
    else:
        thread_pos = fused_spatial_axes[2]

    # always compute at here
    s[write_cache].compute_at(s[op], thread_pos)

    # reduce_split
    reduced_axes = s[write_cache].op.reduce_axis
    splited_reduced_axes = []
    if "reduce" in config and len(config["reduce"]) > 0:
        # to align each axis
        assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
        for axis, nparts in zip(reduced_axes, config["reduce"]):
            tmp_buffer = []
            for count in range(len(nparts) - 1):
                outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_reduced_axes.append(tmp_buffer)
    else:
        for axis in reduced_axes:
            splited_reduced_axes.append([axis])

    # if has reduce axes
    if len(splited_reduced_axes) > 0:
        # always reorder here
        reduced_nonfuse_lsts = []
        reorder_lst = []
        length = len(splited_reduced_axes[0])

        for count in range(length):
            tmp_buffer = [x[count] for x in splited_reduced_axes]
            reduced_nonfuse_lsts.append(tmp_buffer)
            reorder_lst.extend(tmp_buffer)
        # change the order of reduce axes and spatial axes
        rlength = len(splited_reduced_axes)
        if rlength > 1:
            reorder_lst.extend(s[write_cache].op.axis)
        elif rlength == 1:  # in this case, have to interleave otherwise the reorder is of no use
            tmp_order = []
            p_spatial = len(s[write_cache].op.axis) - 1
            p_reduce = len(reorder_lst) - 1
            while p_spatial >= 0 and p_reduce >= 0:
                tmp_order.append(s[write_cache].op.axis[p_spatial])
                tmp_order.append(reorder_lst[p_reduce])
                p_spatial -= 1
                p_reduce -= 1
            while p_spatial >= 0:
                tmp_order.append(s[write_cache].op.axis[p_spatial])
                p_spatial -= 1
            while p_reduce >= 0:
                tmp_order.append(reorder_lst[p_reduce])
                p_reduce -= 1
            tmp_order = list(reversed(tmp_order))
            reorder_lst = tmp_order
        s[write_cache].reorder(*reorder_lst)

    # unroll
    if "unroll" in config and len(config["unroll"]) > 0:
        step = config["unroll"][0][0]
        s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)

    # always vectorize here
    s[write_cache].vectorize(s[write_cache].op.axis[-1])


def cpu_schedule_split_reorder_fuse(config, s, op, op_state):
    # assert_print(op in s)

    loop_idx = []
    loop_lst = []

    # always cache write here
    # if op.num_outputs > 1:
    #     raise RuntimeWarning("Too many outputs in one operation!")
    write_cache = s.cache_write(op.output(0), "local")

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
            tmp_buffer = []
            tmp_extents = []
            for count in range(len(nparts) - 1):
                outer, axis = s[op].split(axis, nparts=nparts[count])
                tmp_buffer.append(outer)
                tmp_extents.append(nparts[count])
            tmp_buffer.append(axis)
            tmp_extents.append(nparts[-1])
            splited_spatial_axes.append(tmp_buffer)
            splited_spatial_extents.append(tmp_extents)
    else:
        for axis in spatial_axes:
            splited_spatial_axes.append([axis])
            splited_spatial_extents.append([axis.dom.extent.value])

    # always reorder here
    reorder_lst = []
    reorder_parts = []
    reorder_part_extents = []
    for count in range(len(splited_spatial_axes[0])):
        tmp_buffer = [x[count] for x in splited_spatial_axes]
        tmp_extents = [x[count] for x in splited_spatial_extents]
        reorder_lst.extend(tmp_buffer)
        reorder_parts.append(tmp_buffer)
        reorder_part_extents.append(tmp_extents)
    s[op].reorder(*reorder_lst)

    # handle fuse request
    fused_parts = []
    fused_part_extents = []
    fused_part_idx = []
    if "fuse" in config and len(config["fuse"]) > 0:
        base_id = 0
        for part, extents in zip(reorder_parts, reorder_part_extents):
            tmp_part = []
            tmp_extents = []
            tmp_idx = []
            idx = 0
            beg = 0
            for end in config["fuse"][0]:
                if end - beg > 1:
                    fuse_lst = part[beg:end]
                    fused = s[op].fuse(*fuse_lst)
                    tmp_part.append(fused)
                    extent = reduce(lambda x, y: x * y, extents[beg:end], 1)
                    tmp_idx.extend([idx] * (end - beg))
                    idx += 1
                    tmp_extents.append(extent)
                elif end - beg == 1:
                    tmp_part.append(part[beg])
                    tmp_extents.append(extents[beg])
                    tmp_idx.append(idx)
                    idx += 1
                beg = end
            fused_parts.append(tmp_part)
            fused_part_extents.append(tmp_extents)
            fused_part_idx.append(tmp_idx)

            # for op state
            loop_lst.extend(tmp_part)
            loop_idx.extend([x + base_id for x in tmp_idx])
            base_id += len(tmp_part)
    else:
        fused_parts = reorder_parts
        fused_part_extents = reorder_part_extents
        fused_part_idx = [list(range(len(x))) for x in reorder_parts]

        # for op state
        loop_lst = reorder_lst
        loop_idx = list(range(len(reorder_lst)))

    # record op state
    op_state.loop_lst = loop_lst
    op_state.loop_idx = loop_idx

    # parallel
    fused = s[op].fuse(*fused_parts[0])
    s[op].parallel(fused)

    # compute at
    num_parts = len(fused_parts)
    if num_parts == 1:
        local_pos = fused
    elif num_parts == 2:
        local_pos = fused_parts[num_parts - 1][0]
    else:
        local_pos = fused_parts[num_parts - 2][-1]

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
        assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
        for axis, nparts in zip(reduced_axes, config["reduce"]):
            tmp_buffer = []
            for count in range(len(nparts) - 1):
                outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_reduced_axes.append(tmp_buffer)
    else:
        for axis in reduced_axes:
            splited_reduced_axes.append([axis])

    # if has reduce axes
    if len(splited_reduced_axes) > 0:
        # always reorder here
        reduced_nonfuse_lsts = []
        reorder_lst = []
        length = len(splited_reduced_axes[0])
        # leave the last part
        for count in range(length - 1):
            tmp_buffer = [x[count] for x in splited_reduced_axes]
            reduced_nonfuse_lsts.append(tmp_buffer)
            reorder_lst.extend(tmp_buffer)
        # the last part
        last_part = [x[length - 1] for x in splited_reduced_axes]
        spatial_remainder = s[write_cache].op.axis
        # change the order of reduce axes and spatial axes
        if "reorder" in config and len(config["reorder"]) > 0:
            pos = config["reorder"][0][0]
            assert pos < len(spatial_remainder)
            tmp_buffer = []
            count = len(spatial_remainder) - 1
            while count > pos:
                tmp_buffer.append(spatial_remainder[count])
                count -= 1
            p = pos
            q = len(last_part) - 1
            while p >= 0 and q >= 0:
                tmp_buffer.append(spatial_remainder[p])
                tmp_buffer.append(last_part[q])
                p -= 1
                q -= 1
            while p >= 0:
                tmp_buffer.append(spatial_remainder[p])
                p -= 1
            while q >= 0:
                tmp_buffer.append(last_part[q])
                q -= 1
            tmp_buffer = list(reversed(tmp_buffer))
            reorder_lst.extend(tmp_buffer)
        else:
            reorder_lst.extend(last_part)
            reorder_lst.extend(spatial_remainder)
        s[write_cache].reorder(*reorder_lst)

    # unroll
    if "unroll" in config and len(config["unroll"]) > 0:
        step = config["unroll"][0][0]
        explicit = config["unroll"][0][1]
        s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        s[op].pragma(kernel_scope, 'unroll_explicit', explicit)


def cpu_schedule_simple(config, s, op, op_state):
    # always cache write here
    # if op.num_outputs > 1:
    #     raise RuntimeWarning("Too many outputs in one operation!")
    write_cache = s.cache_write(op.output(0), "global")
    # spatial split
    spatial_axes = s[op].op.axis
    splited_spatial_axes = []
    if "spatial" in config and len(config["spatial"]) > 0:
        # to align each axis
        assert_print(len(config["spatial"]) == len(spatial_axes), "align failed")
        for axis, nparts in zip(spatial_axes, config["spatial"]):
            nfactors = [1]
            count = len(nparts) - 1
            while count >= 0:
                nfactors.append(nparts[count] * nfactors[-1])
                count -= 1
            tmp_buffer = []
            num_factors = len(nfactors)
            for i in range(num_factors - 2):
                factor = nfactors[num_factors - 2 - i]
                part = nparts[i]
                if factor == 1:
                    tmp_buffer.append(axis)
                    axis = None
                elif part == 1:
                    tmp_buffer.append(None)
                else:
                    outer, axis = s[op].split(axis, factor=factor)
                    tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_spatial_axes.append(tmp_buffer)
    else:
        for axis in spatial_axes:
            splited_spatial_axes.append([axis])
    assert_print(len(splited_spatial_axes) > 0, "empty spatial axes")  # must be non-empty

    # always reorder and fuse here
    # this part actually suppose there is "spatial" in config
    # which is avoidable
    spatial_fuse_lsts = []
    spatial_fuse_extents = []
    reorder_lst = []
    fused_spatial_axes = []
    spatial_split_num_parts = len(splited_spatial_axes[0])
    for count in range(spatial_split_num_parts):
        tmp_buffer = [x[count] for x in splited_spatial_axes]
        tmp_extent = reduce(lambda a, b: a * b, [x[count] for x in config["spatial"]])
        spatial_fuse_lsts.append(tmp_buffer)
        spatial_fuse_extents.append(tmp_extent)
        reorder_lst.extend(tmp_buffer)
    reorder_lst_without_none = list(filter(lambda x: x is not None, reorder_lst))
    # print("reorder op", reorder_lst_without_none)
    s[op].reorder(*reorder_lst_without_none)
    for fuse_lst in spatial_fuse_lsts[:1]:
        tmp_buffer = list(filter(lambda x: x is not None, fuse_lst))
        # print("fuse op", tmp_buffer)
        fused = s[op].fuse(*tmp_buffer)
        fused_spatial_axes.append(fused)
    kernel_scope = fused_spatial_axes[0]
    if len(spatial_fuse_lsts) > 1:
        count = 0
        while count < len(config["spatial"]) and config["spatial"][count][1] == 1:
            count += 1
        if count == len(config["spatial"]):
            count -= 1
        next_pos_for_comptue_at = spatial_fuse_lsts[1][count]
    else:
        next_pos_for_comptue_at = kernel_scope

        # always parallel here
    s[op].parallel(kernel_scope)
    # vectorize
    if len(spatial_fuse_lsts) == 2:
        count = len(spatial_fuse_lsts[1]) - 1
        while count >= 1:
            if spatial_fuse_lsts[1][count] is not None and config["spatial"][1][count] > 1:
                # print("vectorize op", spatial_fuse_lsts[1][count])
                s[op].vectorize(spatial_fuse_lsts[1][count])
                break
            count -= 1
    elif len(spatial_fuse_lsts) > 2:
        count = len(spatial_fuse_lsts[-1]) - 1
        while count >= 0:
            if spatial_fuse_lsts[-1][count] is not None and config["spatial"][count][
                -1] > 1:
                # print("vectorize op", spatial_fuse_lsts[-1][count])
                s[op].vectorize(spatial_fuse_lsts[-1][count])
                break
            count -= 1
    # always compute at here
    # print("compute at", next_pos_for_comptue_at)
    s[write_cache].compute_at(s[op], next_pos_for_comptue_at)

    # spatial_split for write cache
    spatial_axes = s[write_cache].op.axis
    num_spatial_axes = len(spatial_axes)
    splited_spatial_axes = []
    if "spatial" in config and len(config["spatial"]) > 0:
        # to align each axis
        assert_print(len(config["spatial"]) == len(spatial_axes), "align failed")
        for axis, nparts in zip(spatial_axes, config["spatial"]):
            nfactors = [1]
            count = len(nparts) - 1
            while count >= 0:
                nfactors.append(nparts[count] * nfactors[-1])
                count -= 1
            tmp_buffer = []
            num_factors = len(nfactors)
            for i in range(num_factors - 2):
                factor = nfactors[num_factors - 2 - i]
                part = nparts[i]
                if factor == 1:
                    tmp_buffer.append(axis)
                    axis = None
                elif part == 1:
                    tmp_buffer.append(None)
                else:
                    outer, axis = s[write_cache].split(axis, factor=factor)
                    tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_spatial_axes.append(tmp_buffer)
    else:
        for axis in spatial_axes:
            splited_spatial_axes.append([axis])
    assert_print(len(splited_spatial_axes) > 0, "empty spatial axes")  # must be non-empty
    # reduce_split for write cache
    reduced_axes = s[write_cache].op.reduce_axis
    num_reduce_axes = len(reduced_axes)
    splited_reduced_axes = []
    if "reduce" in config and len(config["reduce"]) > 0:
        # to align each axis
        assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
        for axis, nparts in zip(reduced_axes, config["reduce"]):
            nfactors = [1]
            count = len(nparts) - 1
            while count >= 0:
                nfactors.append(nparts[count] * nfactors[-1])
                count -= 1
            tmp_buffer = []
            num_factors = len(nfactors)
            for i in range(num_factors - 2):
                factor = nfactors[num_factors - 2 - i]
                part = nparts[i]
                if factor == 1:
                    tmp_buffer.append(axis)
                    axis = None
                elif part == 1:
                    tmp_buffer.append(None)
                else:
                    outer, axis = s[write_cache].split(axis, factor=factor)
                    tmp_buffer.append(outer)
            tmp_buffer.append(axis)
            splited_reduced_axes.append(tmp_buffer)
    else:
        for axis in reduced_axes:
            splited_reduced_axes.append([axis])
    # for easy align
    # reduce_split_num_parts = len(splited_reduced_axes[0])
    # assert reduce_split_num_parts == spatial_split_num_parts
    # reorder hybrid for spatial and reduce
    hybrid_axes = splited_spatial_axes + splited_reduced_axes
    hybrid_fuse_lsts = []
    hybrid_reorder_lst = []
    for count in range(spatial_split_num_parts):
        tmp_buffer = [x[count] for x in hybrid_axes]
        hybrid_fuse_lsts.append(tmp_buffer)
        hybrid_reorder_lst.extend(tmp_buffer)
    if len(hybrid_fuse_lsts) > 1:
        last_parts = hybrid_reorder_lst[-num_spatial_axes - num_reduce_axes:]
        hybrid_reorder_lst = hybrid_reorder_lst[:-num_spatial_axes - num_reduce_axes]
        tmp_buffer = last_parts[-num_reduce_axes:]
        tmp_buffer.extend(last_parts[:-num_reduce_axes])
        hybrid_reorder_lst.extend(tmp_buffer)
    hybrid_reorder_lst_without_none = list(
        filter(lambda x: x is not None, hybrid_reorder_lst))
    # print("reorder cache write", hybrid_reorder_lst_without_none)
    s[write_cache].reorder(*hybrid_reorder_lst_without_none)
    # fuse without reduce axes
    # assert len(hybrid_fuse_lsts) > 0
    # s[write_cache].fuse(*hybrid_fuse_lsts[0][:-num_reduce_axes])

    # unroll and vectorize without reduce axes
    if len(hybrid_fuse_lsts) > 1:
        rcount = num_spatial_axes - 1
        while rcount >= 0 and config["spatial"][rcount][-1] == 1:
            rcount -= 1
        if rcount >= 0:
            # print("vectorize cache write", hybrid_fuse_lsts[-1][rcount])
            s[write_cache].vectorize(hybrid_fuse_lsts[-1][rcount])
        for count in range(rcount):
            if config["spatial"][count][-1] > 1:
                # print("unroll cache write", hybrid_fuse_lsts[-1][count])
                s[write_cache].unroll(hybrid_fuse_lsts[-1][count])
    if len(hybrid_fuse_lsts) > 2:
        for count in range(num_spatial_axes):
            if config["spatial"][count][-2] > 1:
                # print("unroll cache write", hybrid_fuse_lsts[-2][count])
                s[write_cache].unroll(hybrid_fuse_lsts[-2][count])
        # for count in range(num_reduce_axes):
        #     if config["reduce"][count][-2] > 1:
        #         print("unroll cache write", hybrid_fuse_lsts[-2][count + num_spatial_axes])
        #         s[write_cache].unroll(hybrid_fuse_lsts[-2][count + num_spatial_axes])
