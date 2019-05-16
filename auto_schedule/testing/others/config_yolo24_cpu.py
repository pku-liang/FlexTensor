"""
Configurations of high performance schedules for conv2d_nchw
YOLO convolution layer 24
Target Intel CPU

====================================
**Author**: `Size Zheng`

the terms used below:
------------------------------------
method: fuse_and_bind --> only use parallel
spatial_order: flatten --> bo, ko, po, qo, bm, km, pm, qm, bi, ki, pi, qi...
reduce_order: flatten --> rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi...
spatial_reduce_relation: reduce_before_spatial --> ...rci, ryi, rxi, bi, ki, pi, qi
cache_write_at: last_thread --> compute_at(stage, tx_axis)
"""

yolo_conv24_nchw_b1_cpu = {
    "method": "fuse_and_bind",
    "b_factors": [1, 1, 1],
    "k_factors": [1024, 1, 1],
    "p_factors": [1, 1, 7],
    "q_factors": [1, 1, 7],
    "rc_factors": [1, 16, 64],
    "ry_factors": [1, 1, 3],
    "rx_factors": [1, 1, 3],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [17.2176128, 11.015653610229492],      # [tvm vs. pytorch with MKL enabled]
        "Xeon-E5-2630-v4": [15.639831100000002, 10.886478424072266],
        "Xeon-E5-2690-v4": [11.5555504, 11.779093742370605],
    }
}


yolo_conv24_nchw_b2_cpu = {
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
    }
}


yolo_conv24_nchw_b4_cpu = {
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
    }
}


yolo_conv24_nchw_b8_cpu = {
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
    }
}

yolo_conv24_nchw_b16_cpu = {
    "method": "fuse_and_bind",
    "b_factors": [16, 1, 1],
    "k_factors": [512, 1, 2],
    "p_factors": [1, 1, 7],
    "q_factors": [1, 1, 7],
    "rc_factors": [1, 32, 32],
    "ry_factors": [1, 1, 3],
    "rx_factors": [1, 1, 3],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [287.4641678, 74.42233562469482],      # [tvm vs. pytorch with MKL enabled]
        "Xeon-E5-2630-v4": [202.33267189999998, 47.089266777038574],
        "Xeon-E5-2690-v4": [123.31580509999999, 23.560667037963867],
    }
}


yolo_conv24_nchw_b32_cpu = {
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
    }
}


yolo_conv24_nchw_b64_cpu = {
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
    }
}


yolo_conv24_nchw_b128_cpu = {
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
    }
}


yolo_conv24_nchw_b256_cpu = {
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
    }
}
