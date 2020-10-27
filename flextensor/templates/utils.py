from functools import reduce


def split_axes(config, s, op, axes, axis_type):
    split_axes = []
    split_exts = []
    if axis_type in config and len(config[axis_type]) > 0:
        # to align each axis
        assert len(config[axis_type]) == len(axes), "align failed"
        for axis, nparts_lst in zip(axes, config[axis_type]):
            axes, exts = [], []
            for nparts in nparts_lst[:-1]:
                outer, axis = s[op].split(axis, nparts=nparts)
                axes.append(outer)
                exts.append(nparts)
            axes.append(axis)
            exts.append(nparts_lst[-1])
            split_axes.append(axes)
            split_exts.append(exts)
    else:
        for axis in axes:
            split_axes.append([axis])
            split_exts.append([axis.dom.extent.value])
    return split_axes, split_exts


def fuse_axes(config, s, op, op_state, reorder_parts, reorder_part_exts, reorder_lst):
    fused_parts = []
    fused_part_extents = []
    fused_part_idx = []
    loop_idx = []
    loop_lst = []
    if "fuse" in config and len(config["fuse"]) > 0:
        base_id = 0
        for part, extents in zip(reorder_parts, reorder_part_exts):
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
        fused_part_extents = reorder_part_exts
        fused_part_idx = [list(range(len(x))) for x in reorder_parts]

        loop_lst = reorder_lst
        loop_idx = list(range(len(reorder_lst)))

    # record op state
    op_state.loop_lst = loop_lst
    op_state.loop_idx = loop_idx

    return fused_parts, fused_part_extents, fused_part_idx


def bind_axes(s, op, fused_parts, fused_part_exts, bind_candidate, candidate_exts):
    bind_option = [None, None, None]
    num_parts = len(fused_parts)

    if num_parts == 1:
        bind_option[0] = (fused_parts[0], fused_part_exts[0])
        local_write_pos = fused_parts[0][:len(bind_candidate[0])][-1]
    elif num_parts == 2:
        bind_option[0] = (fused_parts[0], fused_part_exts[0])
        bind_option[2] = (fused_parts[1], fused_part_exts[1])
        local_write_pos = fused_parts[1][:len(bind_candidate[2])][-1]
    else:
        bind_option[0] = (fused_parts[0], fused_part_exts[0])
        bind_option[1] = (fused_parts[1], fused_part_exts[1])
        bind_option[2] = (fused_parts[2], fused_part_exts[2])
        local_write_pos = fused_parts[2][:len(bind_candidate[2])][-1]

    bound_axes = set()
    for option, candidate, extents in zip(bind_option, bind_candidate, candidate_exts):
        if option is not None:
            for i, axis in enumerate(option[0][:len(candidate)]):
                s[op].bind(axis, candidate[i])
                bound_axes.add(axis)
                extents[i] = option[1][i]

    return local_write_pos, bound_axes


def interleave_reorder(config, s, write_cache, spatial_remainder, last_part, reorder_lst):
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
