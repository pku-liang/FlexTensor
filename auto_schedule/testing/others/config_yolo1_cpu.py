"""
Configurations of high performance schedules for conv2d_nchw
YOLO convolution layer 24
Target Intel CPU

====================================
**Author**: `Size Zheng`

the terms used below:
------------------------------------
method: fuse_and_bind --> only use blockIdx.x, vthread.x, threadIdx.x
spatial_order: flatten --> bo, ko, po, qo, bm, km, pm, qm, bi, ki, pi, qi...
reduce_order: flatten --> rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi...
spatial_reduce_relation: reduce_before_spatial --> ...rci, ryi, rxi, bi, ki, pi, qi
cache_write_at: last_thread --> compute_at(stage, tx_axis)
"""

yolo_conv24_nchw_b1_cuda = {
    "method": "fuse_and_bind",
    "b_factors": [1, 1, 1],
    "k_factors": [64, 1, 1],
    "p_factors": [7, 1, 64],
    "q_factors": [14, 1, 32],
    "rc_factors": [1, 1, 3] ,
    "ry_factors": [1, 1, 7],
    "rx_factors": [1, 1, 7],
    "spatial_order": "flatten",
    "reduce_order": "flatten",
    "spatial_reduce_relation": "reduce_before_spatial",
    "cache_write_at": "last_thread",
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [36.860688, 9.055423736572266],      # [tvm vs. pytorch with MKL enabled]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
    }
}

yolo_conv1_nchw_b16_cuda = {
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
        "Core-i7-6700K": [],      # [tvm vs. pytorch with MKL enabled]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
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
    "auto_unroll_max_step": 1500,
    "peformances": {
        "Core-i7-6700K": [],      # [tvm vs. pytorch]
        "Xeon-E5-2630-v4": [],
        "Xeon-E5-2690-v4": [],
    }
}
