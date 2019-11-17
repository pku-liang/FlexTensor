from __future__ import absolute_import 
from auto_schedule.configs.conv2d_config import yolo_shapes


def gflops(batch, in_channel, out_channel, H, W, k_h, k_w, stride, padding, dilation):
    out_h = (H + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (W + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    return 2 * batch * out_h * out_w * in_channel * out_channel * k_h * k_w / 1e9


def perf(gflops, millis):
    return gflops / (millis / 1e3)


if __name__ == "__main__":
    # flextensor
    millis = [
        0.1006952,
        0.2825153,
        0.0252457,
        0.2062096,
        0.0571187,
        0.7426347,
        0.0372696,
        0.210653,
        0.0540586,
        0.7972785,
        0.0652985,
        0.2498188,
        0.5609756,
        0.3801411,
        0.2718407
    ]
    # autotvm
    # millis = [
    #     0.145611,
    #     0.385738,
    #     0.038619,
    #     0.311103,
    #     0.080117,
    #     0.897629,
    #     0.059699,
    #     0.287437,
    #     0.090796,
    #     0.903871,
    #     0.069489,
    #     0.399444,
    #     0.668653,
    #     0.588122,
    #     0.555237
    # ]
    i = 0
    for shape in yolo_shapes:
        batch, in_c, H, W, out_c, _, k_h, k_w, _, stride, padding, dilation, groups = shape 
        print(perf(gflops(batch, in_c, out_c, H, W, k_h, k_w, stride, padding, dilation), millis[i]))
        i += 1
