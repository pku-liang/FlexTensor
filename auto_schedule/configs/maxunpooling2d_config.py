
"""
batch, channels, height, width, kernel_size, stride, padding

(1, 1, 6, 6, 2, 1, 0),
(10, 1, 12, 10, 1, 1, 0), 
(1, 1, 7, 9, 3, 2, 1), 
(1, 1, 16, 16, 4, 4, 2),

DeconvNet --- based on VGG-16
batch size is set to 1
"""
maxunpooling2d_shape = [
    (1, 128, 114, 114, 2, 2, 0),
    (1, 256, 56, 56, 2, 2, 0),
    (1, 512, 28, 28, 2, 2, 0),
    (1, 512, 14, 14, 2, 2, 0)
]