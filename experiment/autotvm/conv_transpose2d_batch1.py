from experiment.util.autotvm_schedule import conv_transpose2d_schedule_dispatcher, conv_transpose2d_evaluate
from experiment.util.autotvm_test import main
from experiment.shape import conv2d_batch1_shapes as shapes
import tvm
from tvm import autotvm
import topi

if __name__ == '__main__':
    name = "conv_transpose2d_batch1"

    def trans(shape):
        N, C, H, W, K, _, Hk, _, _, stride, padding, dilation, _ = shape
        H, W = (H + 2 * padding - dilation * (Hk - 1) - 1) // stride + \
            1, (W + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1
        C, K = K, C
        return (N, C, H, W, K, Hk, stride, padding, dilation)

    main(name, conv_transpose2d_schedule_dispatcher,
         map(trans, shapes), conv_transpose2d_evaluate)
