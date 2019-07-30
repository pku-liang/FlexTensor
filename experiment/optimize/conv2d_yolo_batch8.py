from experiment.util.optimize_test import main
from experiment.shape import conv2d_batch1_shapes as shapes

if __name__ == '__main__':
    def trans(shape):
        _, C, H, W, K, _, Hk, _, _, stride, padding, dilation, groups = shape
        return (8, C, H, W, K, Hk, stride, padding, dilation, groups)
    main('conv2d', __file__[:-3], shapes, trans)
