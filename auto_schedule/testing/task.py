import tvm 
from auto_schedule.testing.ops import conv2d_nchw, gemm as op_gemm, conv1d as op_conv1d, conv3d_ncdhw, \
    gemm_conv2d_nchw, gemv as op_gemv, bilinear as op_bilinear, MTTKRP3d
from auto_schedule.testing.configs.conv1d_config import conv1d_shapes
from auto_schedule.testing.configs.conv2d_config import yolo_shapes, res_shapes, google_shapes, squeeze_shapes, vgg_16_shapes
from auto_schedule.testing.configs.conv3d_config import conv3d_shapes
from auto_schedule.testing.configs.gemv_config import gemv_shapes
from auto_schedule.testing.configs.gemm_config import gemm_shapes
from auto_schedule.testing.configs.bilinear_config import bilinear_shapes
from auto_schedule.testing.configs.mttkrp_config import mttkrp_shapes
from auto_schedule.testing.configs.depthwise_config import depthwise_shapes
from auto_schedule.testing.configs.grouped_config import grouped_shapes
from auto_schedule.testing.configs.dilation_config import dilation_shapes
TASK_TABLE = {}


class Task(object):
    def __init__(self, category, name, func, args, target, dev_id=0):
        self.key = "{}_{}_{}_{}({})".format(category, name, args, target, dev_id)
        self.func = func
        self.args = args
        self.target = target
        self.dev_id = dev_id
        self.category = category


def register_task(task, override=False):
    if task.key in TASK_TABLE and not override:
        raise RuntimeError("Same task occurs again")
    TASK_TABLE[task.key] = task


def conv1d(N, C, L, K, kernel, stride=1, padding=0, dilation=1, groups=1):
    Seq = tvm.placeholder((N, C, L))
    W = tvm.placeholder((K, C//groups, kernel))
    Output = op_conv1d(Seq, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return [Output.op], [Seq, W, Output]


def conv2d(N, C, H, W, K, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    Img = tvm.placeholder((N, C, H, W))
    W = tvm.placeholder((K, C//groups, kernel_size, kernel_size))
    Output = conv2d_nchw(Img, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return [Output.op], [Img, W, Output]


def conv3d(N, C, D, H, W, K, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    Img = tvm.placeholder((N, C, D, H, W))
    W = tvm.placeholder((K, C//groups, kernel_size, kernel_size, kernel_size))
    Output = conv3d_ncdhw(Img, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return [Output.op], [Img, W, Output]


def gemm_conv2d(N, C, H, W, K, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    Img = tvm.placeholder((N, C, H, W))
    W = tvm.placeholder((K, C, kernel_size, kernel_size))
    Output = gemm_conv2d_nchw(Img, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return [Output.op], [Img, W, Output]


def gemv(N, K):
    A = tvm.placeholder((N, K))
    B = tvm.placeholder((K,))
    Output = op_gemv(A, B)
    return [Output.op], [A, B, Output]


def gemm(N, K, M):
    A = tvm.placeholder((N, K))
    B = tvm.placeholder((K, M))
    Output = op_gemm(A, B)
    return [Output.op], [A, B, Output]


def bilinear(N, K1, K2, M):
    A = tvm.placeholder((N, K1))
    B = tvm.placeholder((N, K2))
    C = tvm.placeholder((M, K1, K2))
    Output = op_bilinear(A, B, C)
    return [Output.op], [A, B, C, Output]


def mttkrp(N, K1, K2, M):
    A = tvm.placeholder((N, K1, K2))
    B = tvm.placeholder((K1, M))
    C = tvm.placeholder((K2, M))
    Output = op_bilinear(A, B, C)
    return [Output.op], [A, B, C, Output]


for shape in conv1d_shapes:
    batch, in_channel, length, out_channel, _, k_len, _, stride, padding, _, _ = shape
    for j in range(4):
        register_task(Task("conv1d", "conv1d", conv1d, (batch, in_channel, length, out_channel, k_len, stride, padding, 1, 1), "llvm", j))
        register_task(Task("conv1d", "conv1d", conv1d, (batch, in_channel, length, out_channel, k_len, stride, padding, 1, 1), "cuda", j))


conv2d_shape_dict = {
    "yolo": yolo_shapes,
    "google": google_shapes,
    "res": res_shapes,
    "squeeze": squeeze_shapes,
    "vgg-16": vgg_16_shapes
}

for name in ["yolo", "google", "res", "squeeze", "vgg-16"]:
    shapes = conv2d_shape_dict[name]
    for i, shape in enumerate(shapes):
        batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, _, _ = shape
        for j in range(4):
            register_task(
                Task(
                    "conv2d",
                    name + str(i), 
                    conv2d, 
                    (batch, in_channel, height, width, out_channel, k_h, stride, padding, 1, 1), 
                    "llvm", 
                    j
                    ))
            register_task(
                Task(
                    "conv2d",
                    name + str(i), 
                    conv2d, 
                    (batch, in_channel, height, width, out_channel, k_h, stride, padding, 1, 1), 
                    "cuda", 
                    j
                    ))
            register_task(
                Task(
                    "gemm_conv",
                    name + str(i), 
                    gemm_conv2d, 
                    (batch, in_channel, height, width, out_channel, k_h, stride, padding, 1, 1), 
                    "llvm", 
                    j
                    ))
            register_task(
                Task(
                    "gemm_conv",
                    name + str(i), 
                    gemm_conv2d, 
                    (batch, in_channel, height, width, out_channel, k_h, stride, padding, 1, 1), 
                    "cuda", 
                    j
                    ))

for shape in depthwise_shapes:
    batch, in_channel, H, W, out_channel, k, _, stride, padding, dilation = shape
    for j in range(4):
        register_task(Task("conv2d", "depthwise", conv2d, (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, in_channel), "llvm", j))
        register_task(Task("conv2d", "depthwise", conv2d, (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, in_channel), "cuda", j))

for shape in grouped_shapes:
    batch, in_channel, H, W, out_channel, k, _, stride, padding, dilation, groups = shape
    for j in range(4):
        register_task(Task("conv2d", "grouped", conv2d, (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, groups), "llvm", j))
        register_task(Task("conv2d", "grouped", conv2d, (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, groups), "cuda", j))

for shape in dilation_shapes:
    batch, in_channel, H, W, out_channel, k, _, stride, padding, dilation, groups = shape
    for j in range(4):
        register_task(Task("conv2d", "dilation", conv2d, (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, groups), "llvm", j))
        register_task(Task("conv2d", "dilation", conv2d, (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, groups), "cuda", j))

for shape in conv3d_shapes:
    batch, in_channel, D, H, W, out_channel, _, k, _, stride, padding, _, _ = shape
    for j in range(4):
        register_task(Task("conv3d", "conv3d", conv3d, (batch, in_channel, D, H, W, out_channel, k, stride, padding, 1, 1), "llvm", j))
        register_task(Task("conv3d", "conv3d", conv3d, (batch, in_channel, D, H, W, out_channel, k, stride, padding, 1, 1), "cuda", j))

for shape in gemv_shapes:
    N, K, _ = shape
    for j in range(4):
        register_task(Task("gemv", "gemv", gemv, (N, K), "llvm", j))
        register_task(Task("gemv", "gemv", gemv, (N, K), "cuda", j))

for shape in gemm_shapes:
    N, K, M = shape
    for j in range(4):
        register_task(Task("gemm", "gemm", gemm, (N, K, M), "llvm", j))
        register_task(Task("gemm", "gemm", gemm, (N, K, M), "cuda", j))

for shape in bilinear_shapes:
    N, K, M = shape
    for j in range(4):
        register_task(Task("bilinear", "bilinear", bilinear, (N, K, K, M), "llvm", j))
        register_task(Task("bilinear", "bilinear", bilinear, (N, K, K, M), "cuda", j))

for shape in mttkrp_shapes:
    N, K, M = shape
    for j in range(4):
        register_task(Task("mttkrp", "mttkrp", mttkrp, (N, K, K, M), "llvm", j))
        register_task(Task("mttkrp", "mttkrp", mttkrp, (N, K, K, M), "cuda", j))

