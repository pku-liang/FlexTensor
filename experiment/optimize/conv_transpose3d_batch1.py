from experiment.util.optimize_test import main
from experiment.shape import conv3d_batch1_shapes as shapes

if __name__ == '__main__':
    def trans(shape):
        N, C, D, H, W, K, _, Hk, _, stride, padding, dilation, groups = shape
        D = (D + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1
        H = (H + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1
        W = (W + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1
        K, C = C, K
        return (N, C, D, H, W, K, Hk, stride, padding, dilation, groups)

    main('conv3d', __file__[:-3], shapes, trans)
