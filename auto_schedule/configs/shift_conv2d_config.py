"""
    (N, H, W, C, kernel_size, dilation)
"""
shift_conv2d_shape = [
    # ShiftNet(https://arxiv.org/abs/1801.09392) with input size: 256*256
    # (1, 128, 128, 64, 3, 1), 
    # (1, 128, 128, 64, 3, 1), 
    # (1, 64, 64, 128, 5, 1), 
    (1, 32, 32, 256, 3, 1), 
    # (1, 16, 16, 512, 3, 1)
]