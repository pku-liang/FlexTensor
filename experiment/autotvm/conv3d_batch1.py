from experiment.util.autotvm_schedule import conv3d_schedule_dispatcher, conv3d_evaluate
from experiment.util.autotvm_test import main
from experiment.shape import conv3d_batch1_shapes as shapes
import tvm
from tvm import autotvm
import topi

if __name__ == '__main__':
    name = "conv3d_batch1"

    def trans(shape):
        N, C, D, H, W, K, _, Hk, _, stride, padding, dilation, _ = shape
        return (N, C, D, H, W, K, Hk, stride, padding, dilation)

    main(name, conv3d_schedule_dispatcher, map(trans, shapes), conv3d_evaluate)
