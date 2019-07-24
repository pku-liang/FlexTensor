"""
    (batch, H, W, in_C, out_C, KH, KW, mask_type, bias, dilation, stride, padding)
"""
PixelCNN_shape = [
    (1, 256, 256, 3, 32, 7, 7, 'A', None, 1, 1, 0),
    (1, 128, 128, 256, 256, 3, 3, 'B', None, 1, 1, 0)
]