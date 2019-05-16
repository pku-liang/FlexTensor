"""
Configurations of high performance schedules for conv2d_nchw
YOLO convolution layer 24
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


yolo_conv24_nchw_b1_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [1, 1, 1, 1],
    "k_factors": [64, 2, 8, 1],
    "p_factors": [1, 1, 7, 1],
    "q_factors": [1, 1, 7, 1],
    "rc_factors": [64, 1, 16],
    "ry_factors": [1, 3, 1],
    "rx_factors": [1, 1, 3],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "cache_read_shared_at": "middle_last",
    "cache_read_local_at": "inner_last",
    "auto_unroll_max_step": 1024,
    "unroll_explicit": 1,
    "peformances": {
        "P100": [0.9718367499999999, 1.4515],      # [tvm vs. cudnn]
        "V100": [0.47016555, 0.867193],
        "TITAN_XP": [float("inf"), 1.21669],
        "K20": [1.6183805000000002, 1.2171]
    }
}


yolo_conv24_nchw_b2_cuda = {
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


yolo_conv24_nchw_b4_cuda = {
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


yolo_conv24_nchw_b8_cuda = {
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


yolo_conv24_nchw_b16_cuda = {
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


yolo_conv24_nchw_b32_cuda = {
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
        "P100": [5.46705389, 4.12434],      # [tvm vs. cudnn]
        "V100": [4.06506747, 3.80283],
        "TITAN_XP": [float("inf"), 5.57239],
        "K20": [8.84548961, 5.79628]
    }
}


yolo_conv24_nchw_b64_cuda = {
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


yolo_conv24_nchw_b128_cuda = {
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


yolo_conv24_nchw_b256_cuda = {
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
