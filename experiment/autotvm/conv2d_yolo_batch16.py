from experiment.util.autotvm_schedule import conv2d_schedule_dispatcher, conv2d_evaluate
from experiment.util.autotvm_test import main
from experiment.shape import conv2d_batch1_shapes as shapes
import tvm
from tvm import autotvm
import topi

if __name__ == '__main__':
    name = "conv2d_yolo_batch32"

    def trans(shape):
        _, C, H, W, K, _, Hk, _, _, stride, padding, dilation, _ = shape
        return (16, C, H, W, K, Hk, stride, padding, dilation, 1)

    main(name, conv2d_schedule_dispatcher, map(trans, shapes), conv2d_evaluate)
