from experiment.util.autotvm_schedule import conv_transpose3d_schedule_dispatcher, conv_transpose3d_evaluate
from experiment.util.autotvm_test import main
from experiment.shape import conv3d_batch1_shapes as shapes
import tvm
from tvm import autotvm
import topi

if __name__ == '__main__':
    name = "conv_transpose3d_batch1"

    def trans(shape):
        N, C, D, H, W, K, _, Hk, _, stride, padding, dilation, _ = shape
        D = (D + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1
        H = (H + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1
        W = (W + 2 * padding - dilation * (Hk - 1) - 1) // stride + 1
        K, C = C, K
        return (N, C, D, H, W, K, Hk, stride, padding, dilation)

    main(name, conv_transpose3d_schedule_dispatcher,
         map(trans, shapes), conv_transpose3d_evaluate)
