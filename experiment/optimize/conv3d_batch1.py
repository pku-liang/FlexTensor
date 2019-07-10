from experiment.util.optimize_test import main
from experiment.shape import conv3d_batch1_shapes as shapes

if __name__ == '__main__':
    def trans(shape):
        N, C, D, H, W, K, _, Hk, _, stride, padding, dilation, groups = shape
        return (N, C, D, H, W, K, Hk, stride, padding, dilation, groups)
    main('conv3d', __file__[:-3], shapes, trans)
