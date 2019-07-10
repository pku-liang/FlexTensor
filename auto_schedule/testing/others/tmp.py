"""
autotvm:

('tile_f', [4, 2, 16, 4]), ('tile_y', [7, 2, 1, 1]), 
('tile_x', [1, 1, 14, 1]), ('tile_rc', [256, 4]), 
('tile_ry', [1, 1]), ('tile_rx', [1, 1]), 
('auto_unroll_max_step', 512), 
('unroll_explicit', 1)],,None,2619227


autate:

conv2d_yolo10_(8, 1024, 14, 14, 512, 1, 1, 0, 1, 1)_cuda(1):
[[{
  "fuse": [[1, 2, 4]], 
  "spatial": [[8, 1, 1, 1], [128, 2, 1, 4], [7, 1, 2, 1], [1, 1, 14, 1]], 
  "reduce": [], 
  "reorder": [[2]], 
  "inline": [], 
  "unroll": [[0, 0]], 
  "merge": []
  }, 
  {
    "fuse": [[2, 3, 4]], 
    "spatial": [[2, 4, 1, 1], [8, 4, 8, 2], [7, 1, 2, 1], [1, 1, 14, 1]], 
    "reduce": [[1, 32, 32], [1, 1, 1], [1, 1, 1]], 
    "reorder": [[0]], 
    "inline": [], 
    "unroll": [[1500, 1]], 
    "merge": []
    }], 
    {"fuse": [], "spatial": [], "reduce": [], "reorder": [], "inline": [[1, 0]], "unroll": [], "merge": [[1, 0]]}]


区别：
autotvm在input channel维分了四份[4, 2, 16, 4]，autate分的factor则是[8, 4, 8, 2]
对于batch维，autotvm直接和input channel的最外层循环合并了（fuse），autate是先按[2, 4, 1, 1]分解，然后逐个与input channel维合并
对于output channel维，autotvm的factor是[256, 4]，autate是[1, 32, 32]
"""


# this autate's schedule
# the config parameter can be regarded as:
# {
#     "fuse": [[2, 3, 4]], 
#     "spatial": [[2, 4, 1, 1], [8, 4, 8, 2], [7, 1, 2, 1], [1, 1, 14, 1]], 
#     "reduce": [[1, 32, 32], [1, 1, 1], [1, 1, 1]], 
#     "reorder": [[0]], 
#     "inline": [], 
#     "unroll": [[1500, 1]], 
#     "merge": []
# }
def generate_op_schedule(target, config):
    def _cuda_schedule_split_reorder_fuse(s, op):
        # assert_print(op in s)

        # always cache write here
        if op.num_outputs > 1:
            raise RuntimeWarning("Too many outputs in one operation!")
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
        assert len(spatial_axes) > 0, "empty spatial axes"     # must be non-empty
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
        else:
            fused_parts = reorder_parts
            fused_part_extents = reorder_part_extents
            fused_part_idx = [list(range(len(x))) for x in reorder_parts]
    
        # always bind here
        # - prepare thread axis
        bx = tvm.thread_axis("blockIdx.x")
        by = tvm.thread_axis("blockIdx.y")
        bz = tvm.thread_axis("blockIdx.z")
        vx = tvm.thread_axis("vthread")
        vy = tvm.thread_axis("vthread")
        vz = tvm.thread_axis("vthread")
        tx = tvm.thread_axis("threadIdx.x")
        ty = tvm.thread_axis("threadIdx.y")
        tz = tvm.thread_axis("threadIdx.z")

        blocks = [bz, by, bx]
        threads = [tz, ty, tx]
        vthreads = [vz, vy, vx]

        block_extents = [-1, -1, -1]    # z, y, x
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




# this is autotvm schedule
# the config parameter can be regarded as:
# [('tile_f', [4, 2, 16, 4]), ('tile_y', [7, 2, 1, 1]), 
# ('tile_x', [1, 1, 14, 1]), ('tile_rc', [256, 4]), 
# ('tile_ry', [1, 1]), ('tile_rx', [1, 1]), 
# ('auto_unroll_max_step', 512), 
# ('unroll_explicit', 1)]

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