from experiment.util.autotvm_schedule import conv2d_schedule_dispatcher, conv2d_evaluate
from experiment.util.autotvm_test import main
from experiment.shape import group_conv2d_batch1_shapes as shapes
from auto_schedule.testing.task import conv2d
import tvm
from tvm import autotvm
import topi

if __name__ == '__main__':
    name = "group_conv2d_batch1"

    def trans(shape):
        N, C, H, W, K, _, Hk, _, _, stride, padding, dilation, groups = shape
        return (N, C, H, W, K, Hk, stride, padding, dilation, groups)

    main(name, conv2d_schedule_dispatcher, map(trans, shapes), conv2d_evaluate)
