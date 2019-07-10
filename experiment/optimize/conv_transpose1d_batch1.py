from experiment.util.optimize_test import main
from experiment.shape import conv1d_batch1_shapes as shapes

if __name__ == '__main__':
    def trans(shape):
        N, C, L, K, _, Lk, _, stride, padding, dilation, groups = shape
        L = (L + 2 * padding - dilation * (Lk - 1) - 1) // stride + 1
        C, K = K, C
        return (N, C, L, K, Lk, stride, padding, dilation, groups)
    main('conv1d', __file__[:-3], shapes, trans)
