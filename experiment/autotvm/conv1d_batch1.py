from experiment.util.autotvm_schedule import conv1d_schedule_dispatcher, conv1d_evaluate
from experiment.util.autotvm_test import main
from experiment.shape import conv1d_batch1_shapes as shapes
import tvm
from tvm import autotvm
import topi

if __name__ == '__main__':
    name = "conv1d_batch1"

    def trans(shape):
        N, C, L, K, _, Lk, _, stride, padding, dilation, _ = shape
        return (N, C, L, K, Lk, stride, padding, dilation)

    main(name, conv1d_schedule_dispatcher, map(trans, shapes), conv1d_evaluate)
