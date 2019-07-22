from experiment.util.autotvm_schedule import conv2d_cuda_schedule, conv2d_evaluate
from experiment.util.autotvm_test import run
from experiment.shape import conv2d_yolo_batch32_shapes as shapes
from auto_schedule.testing.task import conv2d
import tvm
from tvm import autotvm
import topi

if __name__ == '__main__':
    name = "conv2d_yolo_batch32"
    for shape in shapes:
        N, C, H, W, K, _, Hk, _, _, stride, padding, dilation, _ = shape
        args = (N, C, H, W, K, Hk, stride, padding, dilation)
        run(name, conv2d_cuda_schedule, args,
            conv2d_evaluate(*args), target='cuda', trials=200)
