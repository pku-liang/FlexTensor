from experiment.util.autotvm_test import run
from auto_schedule.testing.task import conv2d
import tvm
from tvm import autotvm
import topi
import time


def schedule_llvm(cfg, s, conv):
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    

@autotvm.template
def conv2d_template(N, C, H, W, K, Hk, stride, padding, dilation, groups, target):
    _, [data, kernel, conv] = conv2d(N, C, H, W, K, Hk, stride=stride,
                                     padding=padding, dilation=dilation, groups=groups)

    s = tvm.create_schedule([conv.op])
    cfg = autotvm.get_config()

    return s, [data, kernel, conv]
