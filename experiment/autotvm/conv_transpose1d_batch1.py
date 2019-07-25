from experiment.util.autotvm_schedule import conv_transpose1d_schedule_dispatcher, conv_transpose1d_evaluate
from experiment.util.autotvm_test import main
from experiment.shape import conv1d_batch1_shapes as shapes
import tvm
from tvm import autotvm
import topi

if __name__ == '__main__':
    name = "conv_transpose1d_batch1"

    def trans(shape):
        N, C, L, K, _, Lk, _, stride, padding, dilation, _ = shape
        L = (L + 2 * padding - dilation * (Lk - 1) - 1) // stride + 1
        C, K = K, C
        return (N, C, L, K, Lk, stride, padding, dilation)

    main(name, conv_transpose1d_schedule_dispatcher,
         map(trans, shapes), conv_transpose1d_evaluate)
