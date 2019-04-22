"""
Configurations of high performance schedules for conv2d_nchw
YOLO convolution layer 1
Target NVIDIA GPU

====================================
**Author**: `Size Zheng`

the terms used below:
------------------------------------
method: fuse_and_bind --> only use blockIdx.x, vthread.x, threadIdx.x
spatial_order: flatten --> bo, ko, po, qo, bm, km, pm, qm, bi, ki, pi, qi...
reduce_order: flatten --> rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi...
spatial_reduce_relation: reduce_before_spatial --> ...rci, ryi, rxi, bi, ki, pi, qi
cache_write_at: last_thread --> compute_at(stage, tx_axis)
cache_read_shared_at: milddle_last --> rxm
cache_read_local_at: inner_last --> rxi
"""


yolo_conv1_nchw_b1_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [1, 1, 1, 1],
    "k_factors": [4, 4, 2, 2],
    "p_factors": [14, 2, 8, 2],
    "q_factors": [7, 4, 8, 2],
    "rc_factors": [1, 3, 1],
    "ry_factors": [1, 1, 7],
    "rx_factors": [1, 1, 7],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "cache_read_shared_at": "middle_last",
    "cache_read_local_at": "inner_last",
    "auto_unroll_max_step": 1024,
    "unroll_explicit": 1,
    "peformances": {
        "P100": [0.27035575, 0.866507],      # [tvm vs. cudnn]
        "V100": [0.12002815, 0.13868],
        "TITAN_XP": [float("+inf"), 0.885483],
        "K20": [3.10213688, 0.856076]
    }
}


yolo_conv1_nchw_b2_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [],
    "k_factors": [],
    "p_factors": [],
    "q_factors": [],
    "rc_factors": [],
    "ry_factors": [],
    "rx_factors": [],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "cache_read_shared_at": "middle_last",
    "cache_read_local_at": "inner_last",
    "auto_unroll_max_step": 1024,
    "unroll_explicit": 1,
    "peformances": {
        "P100": [],      # [tvm vs. cudnn]
        "V100": [],
        "TITAN_XP": [],
        "K20": []
    }
}


yolo_conv1_nchw_b4_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [],
    "k_factors": [],
    "p_factors": [],
    "q_factors": [],
    "rc_factors": [],
    "ry_factors": [],
    "rx_factors": [],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "cache_read_shared_at": "middle_last",
    "cache_read_local_at": "inner_last",
    "auto_unroll_max_step": 1024,
    "unroll_explicit": 1,
    "peformances": {
        "P100": [],      # [tvm vs. cudnn]
        "V100": [],
        "TITAN_XP": [],
        "K20": []
    }
}


yolo_conv1_nchw_b8_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [4, 1, 1, 1],
    "k_factors": [1, 4, 8, 2],
    "p_factors": [14, 2, 8, 2],
    "q_factors": [14, 2, 8, 2],
    "rc_factors": [3, 1, 1],
    "ry_factors": [1, 1, 7],
    "rx_factors": [1, 1, 7],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "cache_read_shared_at": "middle_last",
    "cache_read_local_at": "inner_last",
    "auto_unroll_max_step": 1024,
    "unroll_explicit": 1,
    "peformances": {
        "P100": [1.73013305, 5.57216],      # [tvm vs. cudnn]
        "V100": [0.9242939299999999, 1.48438],
        "TITAN_XP": [float("+inf"), 6.03721],
        "K20": [12.052799679999998, 6.2009]
    }
}


yolo_conv1_nchw_b16_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [2, 2, 4, 1],
    "k_factors": [4, 2, 4, 2],
    "p_factors": [14, 2, 8, 2],
    "q_factors": [14, 2, 8, 2],
    "rc_factors": [3, 1, 1],
    "ry_factors": [1, 1, 7],
    "rx_factors": [1, 1, 7],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "cache_read_shared_at": "middle_last",
    "cache_read_local_at": "inner_last",
    "auto_unroll_max_step": 1024,
    "unroll_explicit": 1,
    "peformances": {
        "P100": [4.55509603, 11.0041],      # [tvm vs. cudnn]
        "V100": [3.1160153299999998, 3.16361],
        "TITAN_XP": [float("+inf"), 10.3647],
        "K20": [6.72569161, 10.3739]
    }
}


yolo_conv1_nchw_b32_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [],
    "k_factors": [],
    "p_factors": [],
    "q_factors": [],
    "rc_factors": [],
    "ry_factors": [],
    "rx_factors": [],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "cache_read_shared_at": "middle_last",
    "cache_read_local_at": "inner_last",
    "auto_unroll_max_step": 1024,
    "unroll_explicit": 1,
    "peformances": {
        "P100": [],      # [tvm vs. cudnn]
        "V100": [],
        "TITAN_XP": [],
        "K20": []
    }
}


yolo_conv1_nchw_b64_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [],
    "k_factors": [],
    "p_factors": [],
    "q_factors": [],
    "rc_factors": [],
    "ry_factors": [],
    "rx_factors": [],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "cache_read_shared_at": "middle_last",
    "cache_read_local_at": "inner_last",
    "auto_unroll_max_step": 1024,
    "unroll_explicit": 1,
    "peformances": {
        "P100": [],      # [tvm vs. cudnn]
        "V100": [],
        "TITAN_XP": [],
        "K20": []
    }
}


yolo_conv1_nchw_b128_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [],
    "k_factors": [],
    "p_factors": [],
    "q_factors": [],
    "rc_factors": [],
    "ry_factors": [],
    "rx_factors": [],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "cache_read_shared_at": "middle_last",
    "cache_read_local_at": "inner_last",
    "auto_unroll_max_step": 1024,
    "unroll_explicit": 1,
    "peformances": {
        "P100": [],      # [tvm vs. cudnn]
        "V100": [],
        "TITAN_XP": [],
        "K20": []
    }
}


yolo_conv1_nchw_b256_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [],
    "k_factors": [],
    "p_factors": [],
    "q_factors": [],
    "rc_factors": [],
    "ry_factors": [],
    "rx_factors": [],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "cache_read_shared_at": "middle_last",
    "cache_read_local_at": "inner_last",
    "auto_unroll_max_step": 1024,
    "unroll_explicit": 1,
    "peformances": {
        "P100": [],      # [tvm vs. cudnn]
        "V100": [],
        "TITAN_XP": [],
        "K20": []
    }
}