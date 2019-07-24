"""
    (N, H, W, C, kernel_size, dilation)
    all the shapes bellow are based on ShiftNet and ImageNet dataset cropped to 256*256
    batch size is set to 1
"""
shift_conv2d_shape = [
    (1, 128, 128, 64, 3, 1), 
    (1, 128, 128, 64, 3, 1), 
    (1, 64, 64, 128, 5, 1), 
    (1, 32, 32, 256, 3, 1), 
    (1, 16, 16, 512, 3, 1)
]