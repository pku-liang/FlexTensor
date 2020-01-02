import tvm
import math
import torch
import numpy as np
from flextensor.nn import ShiftConv2d_nhwc

shift_conv2d_shape = [
    # ShiftNet(https://arxiv.org/abs/1801.09392) with input size: 256*256
    (1, 128, 128, 64, 3, 1), 
    (1, 128, 128, 64, 3, 1), 
    (1, 64, 64, 128, 5, 1), 
    (1, 32, 32, 256, 3, 1), 
    (1, 16, 16, 512, 3, 1)
]

DEV_ID = 0


def schedule_shift_1_x86(s, Img, KernelIndex, Output):
    return


def evaluate(shape, schedule_func):
    N, H, W, C, k, dilation = shape
    stride = 1
    Img = tvm.placeholder([N, H, W, C], dtype="float32")
    KernelIndex = tvm.placeholder([C], dtype="int32")
    Output = ShiftConv2d_nhwc(Img, KernelIndex, k, dilation, stride)

    s = tvm.create_schedule(Output.op)
    schedule_func(s, Img, KernelIndex, Output)

    func = tvm.build(s, [Img, KernelIndex, Output], "llvm")
    Img_torch = torch.rand([N, H, W, C], dtype=torch.float32)
    Kernel_torch = torch.rand([C, k, k], dtype=torch.float32)
    KernelIndex_torch = torch.argmax(Kernel_torch.reshape([C, -1]), dim=1)

    paddings = [math.ceil(((stride - 1) * H - stride + dilation * (k - 1)) / 2), 
                math.ceil(((stride - 1) * W - stride + dilation * (k - 1)) / 2)]
    image_height = H
    image_width = W
    out_height = math.floor((image_height + 2 * paddings[0]- dilation * (k - 1) - 1) / stride + 1)
    out_width = math.floor((image_width + 2 * paddings[1] - dilation * (k - 1) - 1) / stride + 1)
    output_shape = (N, out_height, out_width, C)

    Output_torch = torch.zeros(output_shape, dtype=torch.float32)

    ctx = tvm.context("llvm", DEV_ID)

    Img_tvm = tvm.nd.array(Img_torch.numpy().astype(np.float32), ctx)
    KernelIndex_tvm = tvm.nd.array(KernelIndex_torch.numpy().astype(np.int32), ctx)
    Output_tvm = tvm.nd.array(Output_torch.numpy().astype(np.float32), ctx)


    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    time_cost = evaluator(Img_tvm, KernelIndex_tvm, Output_tvm).mean * 1e3

    return time_cost



def main():
    print(evaluate(shift_conv2d_shape[0], schedule_shift_1_x86))


if __name__ == "__main__":
    main()