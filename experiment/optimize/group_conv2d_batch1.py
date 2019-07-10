from experiment.util.optimize_test import main
from experiment.shape import group_conv2d_batch1_shapes as shapes

if __name__ == '__main__':
    def trans(shape):
        N, C, H, W, K, Hk, _, stride, padding, dilation, groups = shape
        return (N, C, H, W, K, Hk, stride, padding, dilation, groups)
    main('conv2d', __file__[:-3], shapes, trans)
