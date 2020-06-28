import tvm 

from flextensor.nn import conv2d_nchw, gemm as op_gemm, conv1d as op_conv1d, conv3d_ncdhw, \
    gemm_conv2d_nchw, gemv as op_gemv, bilinear as op_bilinear, MTTKRP3d, conv_transpose1d as op_conv_transpose1d, \
    conv_transpose2d_nchw, conv_transpose3d_ncdhw, depthwise_conv2d_nchw, block_circulant_matrix as op_block_circulant_matrix, \
    PixelCNN as op_pixel_cnn, GatedPixelCNN as op_gated_pixel_cnn, MaxUnpooling1d as op_maxunpool1d, MaxUnpooling2d as op_maxunpool2d, \
    ShiftConv2d_nhwc as op_shift_conv2d, conv2d_nchwc

from flextensor.configs.conv1d_config import conv1d_shapes
from flextensor.configs.conv2d_config import yolo_shapes, res_shapes, google_shapes, squeeze_shapes, \
    vgg_16_shapes, test_conv_shapes, yolo_shapes_b8, mobilev2_shapes, overfeat_shapes
from flextensor.configs.conv3d_config import conv3d_shapes
from flextensor.configs.gemv_config import gemv_shapes
from flextensor.configs.gemm_config import gemm_shapes, test_gemm_shapes
from flextensor.configs.bilinear_config import bilinear_shapes
from flextensor.configs.mttkrp_config import mttkrp_shapes
from flextensor.configs.depthwise_config import depthwise_shapes
from flextensor.configs.grouped_config import grouped_shapes
from flextensor.configs.dilation_config import dilation_shapes
from flextensor.configs.block_circulant_matrix_config import block_circulant_matrix_shapes
from flextensor.configs.maxunpooling1d_config import maxunpooling1d_shape
from flextensor.configs.maxunpooling2d_config import maxunpooling2d_shape
from flextensor.configs.PixelCNN_config import PixelCNN_shape
from flextensor.configs.gated_pixelcnn_config import gated_pixelcnn_shape
from flextensor.configs.shift_conv2d_config import shift_conv2d_shape

from flextensor.space import EnumSpace

TASK_TABLE = {}


class Task(object):
    def __init__(self, category, name, func, args, target, dev_id=0):
        self.key = "{}_{}_{}_{}({})".format(category, name, args, target, dev_id)#
        self.func = func
        self.args = args
        self.target = target
        self.dev_id = dev_id
        self.category = category
        self.special_space = {}

    def set_specific_space(self, key, type, knobs):
        if type == "enum":
            self.special_space[key] = EnumSpace(knobs)
        else:
            raise RuntimeError("Not supported space type: %s" % type)


def register_task(task, override=False):
    if task.key in TASK_TABLE and not override:
        pass # print("[Warning]: Same task occurs again %s" % task.key)
    TASK_TABLE[task.key] = task


def register(func, category, name, args, target, dev_id=0, override=False):
    task = Task(category, name, func, args, target, dev_id)
    register_task(task, override=override)


def conv1d(N, C, L, K, kernel, stride=1, padding=0, dilation=1, groups=1):
    Seq = tvm.placeholder((N, C, L))
    W = tvm.placeholder((K, C//groups, kernel))
    Output = op_conv1d(Seq, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return [Output.op], [Seq, W, Output]


def conv_transpose1d(N, C, L, K, kernel, stride=1, padding=0, dilation=1, groups=1):
    Seq = tvm.placeholder((N, C, L))
    W = tvm.placeholder((C, K//groups, kernel))
    Output = op_conv_transpose1d(Seq, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return [Output.op], [Seq, W, Output]


def conv2d(N, C, H, W, K, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    Img = tvm.placeholder((N, C, H, W))
    W = tvm.placeholder((K, C//groups, kernel_size, kernel_size))
    Output = conv2d_nchw(Img, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return [Output.op], [Img, W, Output]


def conv2d_nchwc_layout(N, C, H, W, K, k=3, st=1, pad=0, dilation=1, group=1, vlen1=8, vlen2=8):
    use_bias = False
    inputs = tvm.placeholder([N, C // vlen1 // group, H, W, vlen1], dtype="float32")
    weight = tvm.placeholder([K // vlen2, C // vlen1 // group, k, k, vlen1, vlen2], dtype="float32")
    if use_bias:
        bias = tvm.placeholder([K // vlen2, vlen2], dtype="float32")
    else:
        bias = None 
    output = conv2d_nchwc(inputs, weight, bias, stride=st, padding=pad, dilation=dilation, groups=group)
    if use_bias:
        return [output.op], [inputs, weight, bias, output]
    else:
        return [output.op], [inputs, weight, output]


def depthwise_conv2d(N, C, H, W, factor, kernel_size, stride=1, padding=0, dilation=1):
    Img = tvm.placeholder((N, C, H, W))
    W = tvm.placeholder((C, factor, kernel_size, kernel_size))
    Ouput = depthwise_conv2d_nchw(Img, W, stride=stride, padding=padding, dilation=dilation)
    return [Ouput.op], [Img, W, Ouput]


def conv_transpose2d(N, C, H, W, K, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1):
    Inputs = tvm.placeholder((N, C, H, W))
    W = tvm.placeholder((C, K//groups, kernel_size, kernel_size))
    Output = conv_transpose2d_nchw(Inputs, W, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups)
    return [Output.op], [Inputs, W, Output]


def conv3d(N, C, D, H, W, K, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    Img = tvm.placeholder((N, C, D, H, W))
    W = tvm.placeholder((K, C//groups, kernel_size, kernel_size, kernel_size))
    Output = conv3d_ncdhw(Img, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return [Output.op], [Img, W, Output]


def conv_transpose3d(N, C, D, H, W, K, kernel_size, stride=1, padding=0, dialtion=1, groups=1):
    Img = tvm.placeholder((N, C, D, H, W))
    W = tvm.placeholder((C, K//groups, kernel_size, kernel_size, kernel_size))
    Output = conv_transpose3d_ncdhw(Img, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
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


def gemm(N, K, M, dtype="float32"):
    A = tvm.placeholder((N, K), dtype=dtype)
    B = tvm.placeholder((K, M), dtype=dtype)
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
    Output = MTTKRP3d(A, B, C)
    return [Output.op], [A, B, C, Output]


def conv2d_1x1_packed(N, C, H, W, K, kernel_size):
    A = tvm.placeholder((N, C, H * W))
    B = tvm.placeholder((K, C))
    rc = tvm.reduce_axis((0, C))
    Output = tvm.compute((N, K, H * W), lambda b, k, i: tvm.sum(A[b, rc, i] * B[k, rc], axis=rc))
    return [Output.op], [A, B, Output]

def block_circulant_matrix(ROW, COL, FFT):
    Input = tvm.placeholder((ROW, COL))
    Output = op_block_circulant_matrix(Input, FFT)
    return [Output.op], [Input, Output]

def maxunpooling1d(N, C, L, kernel_size, stride=1, padding=0):
    Input = tvm.placeholder((N, C, L))
    Indices = tvm.placeholder((N, C, L))
    Output = op_maxunpool1d(Input, Indices, kernel_size, stride=stride, padding=padding)
    return [Output.op], [Input, Indices, Output]

def maxunpooling2d(N, C, H, W, kernel_size, stride=1, padding=0):
    Input = tvm.placeholder((N, C, H, W))
    Indices = tvm.placeholder((N, C, H, W))
    Output = op_maxunpool2d(Input, Indices, kernel_size, stride=stride, padding=padding)
    return [Output.op], [Input, Indices, Output]

def shiftconv2d(N, H, W, C, kernel_size, dialtion=1, stride=1):
    Input = tvm.placeholder((N, H, W, C))
    KernelIndex = tvm.placeholder((C, ), dtype="int32")
    # PInput, kernelIndex, Output = op_shift_conv2d(Input, Kernel, dilation, stride)
    # return [PInput.op, kernelIndex.op, Output.op], [Input, Kernel, Output]
    Output = op_shift_conv2d(Input, KernelIndex, kernel_size, dilation, stride)
    return [Output.op], [Input, KernelIndex, Output]

def pixelcnn(N, H, W, C, OutC, kernel_height, kernel_width, mask_type, bias=None, stride=1, padding=0, dilation=1, groups=1):
    Input = tvm.placeholder((N, H, W, C))
    Kernel = tvm.placeholder((OutC, C, kernel_height, kernel_width))
    Mask, Output = op_pixel_cnn(Input, Kernel, mask_type, bias=bias, dilation=dilation, stride=stride, padding=padding)
    return [Mask.op, Output.op], [Input, Kernel, Mask, Output]

def gatedpixelcnn(N, H, W, C, OutC, kernel_size, ClassVector=None, bias=None, stride=1, padding=0, dilation=1, groups=1):
    Input = tvm.placeholder((N, H, W, C))
    KernelV = tvm.placeholder((2 * OutC, C, kernel_size, kernel_size))
    KernelV2H = tvm.placeholder((2 * OutC, 2 * OutC, 1, 1))
    KernelH = tvm.placeholder((2 * OutC, C, 1, kernel_size))
    KernelHOut = tvm.placeholder((OutC, OutC, 1, 1))
    if ClassVector is not None:
        ClassVector = tvm.placeholder((N, 2 * OutC, 1, 1))
    GateV, Output = op_gated_pixel_cnn(Input, KernelV, KernelV2H, KernelH, KernelHOut, ClassVector, bias=bias, dilation=dilation, stride=stride, padding=padding)
    return [GateV.op, Output.op], [Input, KernelV, KernelV2H, KernelH, KernelHOut, GateV, Output]


# register_task(Task("conv2d", "1x1-packed", conv2d_1x1_packed, (256, 256, 14, 14, 512, 1), "cuda", 0))


for shape in conv1d_shapes:
    batch, in_channel, length, out_channel, _, k_len, _, stride, padding, dilation, groups = shape
    rin_channel = out_channel
    rout_channel = in_channel
    rlength = (length + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1
    for j in range(4):
        register_task(
            Task(
                "conv1d", 
                "conv1d", 
                conv1d, 
                (batch, in_channel, length, out_channel, k_len, stride, padding, dilation, groups), 
                "llvm", 
                j
                ))
        register_task(
            Task(
                "conv1d", 
                "conv1d", 
                conv1d, 
                (batch, in_channel, length, out_channel, k_len, stride, padding, dilation, groups), 
                "cuda", 
                j
                ))
        register_task(
            Task(
                "conv_transpose1d", 
                "conv_transpose1d", 
                conv_transpose1d, 
                (batch, rin_channel, rlength, rout_channel, k_len, stride, padding, dilation, groups), 
                "llvm", 
                j
                ))
        register_task(
            Task(
                "conv_transpose1d", 
                "conv_transpose1d", 
                conv_transpose1d, 
                (batch, rin_channel, rlength, rout_channel, k_len, stride, padding, dilation, groups), 
                "cuda", 
                j
                ))


conv2d_shape_dict = {
    "yolo": yolo_shapes,
    "google": google_shapes,
    "res": res_shapes,
    "squeeze": squeeze_shapes,
    "vgg-16": vgg_16_shapes,
    "test": test_conv_shapes,
    "yolo_b8": yolo_shapes_b8,
    "mobile_v2": mobilev2_shapes,
    "overfeat": overfeat_shapes
}

for name in ["yolo", "google", "res", "squeeze", "vgg-16", "test", "yolo_b8", "mobile_v2", "overfeat"]:
    shapes = conv2d_shape_dict[name]
    for i, shape in enumerate(shapes):
        batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
        rout_channel = in_channel
        rin_channel = out_channel
        rheight = (height + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
        rwidth = (width + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
        for j in range(4):
            register_task(
                Task(
                    "conv2d",
                    name + str(i), 
                    conv2d, 
                    (batch, in_channel, height, width, out_channel, k_h, stride, padding, dilation, groups), 
                    "llvm", 
                    j
                    ))
            register_task(
                Task(
                    "conv2d",
                    name + str(i), 
                    conv2d, 
                    (batch, in_channel, height, width, out_channel, k_h, stride, padding, dilation, groups), 
                    "cuda", 
                    j
                    ))
            register_task(
                Task(
                    "gemm_conv2d",
                    name + str(i), 
                    gemm_conv2d, 
                    (batch, in_channel, height, width, out_channel, k_h, stride, padding, dilation, groups), 
                    "llvm", 
                    j
                    ))
            register_task(
                Task(
                    "gemm_conv2d",
                    name + str(i), 
                    gemm_conv2d, 
                    (batch, in_channel, height, width, out_channel, k_h, stride, padding, dilation, groups), 
                    "cuda", 
                    j
                    ))
            register_task(
                Task(
                    "conv_transpose2d",
                    name + str(i), 
                    conv_transpose2d, 
                    (batch, rin_channel, rheight, rwidth, rout_channel, k_h, stride, padding, dilation, groups), 
                    "llvm", 
                    j
                    ))
            register_task(
                Task(
                    "conv_transpose2d",
                    name + str(i), 
                    conv_transpose2d, 
                    (batch, rin_channel, rheight, rwidth, rout_channel, k_h, stride, padding, dilation, groups), 
                    "cuda", 
                    j
                    ))
            # register common conv2d
            register_task(
                Task(
                    "conv2d",
                    "conv2d", 
                    conv2d, 
                    (batch, in_channel, height, width, out_channel, k_h, stride, padding, dilation, groups), 
                    "llvm", 
                    j
                    ))
            register_task(
                Task(
                    "conv2d",
                    "conv2d", 
                    conv2d, 
                    (batch, in_channel, height, width, out_channel, k_h, stride, padding, dilation, groups), 
                    "cuda", 
                    j
                    ))

for shape in depthwise_shapes:
    batch, in_channel, H, W, factor, k, _, stride, padding, dilation = shape
    for j in range(4):
        register_task(
            Task(
                "conv2d", 
                "depthwise", 
                depthwise_conv2d, 
                (batch, in_channel, H, W, factor, k, stride, padding, dilation), 
                "llvm", 
                j
                ))
        register_task(
            Task(
                "conv2d", 
                "depthwise", 
                depthwise_conv2d, 
                (batch, in_channel, H, W, factor, k, stride, padding, dilation), 
                "cuda", 
                j
                ))

for shape in grouped_shapes:
    batch, in_channel, H, W, out_channel, k, _, stride, padding, dilation, groups = shape
    for j in range(4):
        register_task(
            Task(
                "conv2d", 
                "grouped", 
                conv2d, 
                (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, groups), 
                "llvm", 
                j
                ))
        register_task(
            Task(
                "conv2d", 
                "grouped", 
                conv2d, 
                (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, groups), 
                "cuda", 
                j
                ))

for shape in dilation_shapes:
    batch, in_channel, H, W, out_channel, k, _, stride, padding, dilation, groups = shape
    for j in range(4):
        register_task(
            Task(
                "conv2d", 
                "dilation", 
                conv2d, 
                (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, groups), 
                "llvm", 
                j
                ))
        register_task(
            Task(
                "conv2d", 
                "dilation", 
                conv2d, 
                (batch, in_channel, H, W, out_channel, k, stride, padding, dilation, groups), 
                "cuda", 
                j
                ))

for shape in conv3d_shapes:
    batch, in_channel, D, H, W, out_channel, _, k, _, stride, padding, dilation, groups = shape
    rin_channel = out_channel
    rout_channel = in_channel
    rD = (D + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    rH = (H + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    rW = (W + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    for j in range(4):
        register_task(
            Task(
                "conv3d", 
                "conv3d", 
                conv3d, 
                (batch, in_channel, D, H, W, out_channel, k, stride, padding, dilation, groups), 
                "llvm", 
                j
                ))
        register_task(
            Task(
                "conv3d", 
                "conv3d", 
                conv3d, 
                (batch, in_channel, D, H, W, out_channel, k, stride, padding, dilation, groups), 
                "cuda", 
                j
                ))
        register_task(
            Task(
                "conv_transpose3d", 
                "conv_transpose3d", 
                conv_transpose3d, 
                (batch, rin_channel, rD, rH, rW, rout_channel, k, stride, padding, dilation, groups), 
                "llvm", 
                j
                ))
        register_task(
            Task(
                "conv_transpose3d", 
                "conv_transpose3d", 
                conv_transpose3d, 
                (batch, rin_channel, rD, rH, rW, rout_channel, k, stride, padding, dilation, groups), 
                "cuda", 
                j
                ))

for shape in gemv_shapes:
    N, K, _ = shape
    for j in range(4):
        register_task(Task("gemv", "gemv", gemv, (N, K), "llvm", j))
        register_task(Task("gemv", "gemv", gemv, (N, K), "cuda", j))

for shape in gemm_shapes:
    N, K, M = shape
    for j in range(4):
        register_task(Task("gemm", "gemm", gemm, (N, K, M, "int8"), "micro", j))
        register_task(Task("gemm", "gemm", gemm, (N, K, M, "float32"), "llvm", j))
        register_task(Task("gemm", "gemm", gemm, (N, K, M, "float32"), "cuda", j))

# for shape in test_gemm_shapes:
#     N, K, M = shape
#     for j in range(4):
#         register_task(Task("gemm", "test_gemm", gemm, (N, K, M), "llvm", j))
#         register_task(Task("gemm", "test_gemm", gemm, (N, K, M), "cuda", j))

for shape in bilinear_shapes:
    N, K1, K2, M = shape
    for j in range(4):
        register_task(Task("bilinear", "bilinear", bilinear, (N, K1, K2, M), "llvm", j))
        register_task(Task("bilinear", "bilinear", bilinear, (N, K1, K2, M), "cuda", j))

for shape in mttkrp_shapes:
    N, K1, K2, M = shape
    for j in range(4):
        register_task(Task("mttkrp", "mttkrp", mttkrp, (N, K1, K2, M), "llvm", j))
        register_task(Task("mttkrp", "mttkrp", mttkrp, (N, K1, K2, M), "cuda", j))

for shape in block_circulant_matrix_shapes:
    ROW, COL, FFT = shape
    for j in range(4):
        for platform in ('llvm', 'cuda'):
            register_task(Task('block_circulant_matrix', 'block_circulant_matrix', block_circulant_matrix, (ROW, COL, FFT), platform, j))

for shape in maxunpooling1d_shape:
    for j in range(4):
        register_task(Task("maxunpooling1d", "maxunpooling1d", maxunpooling1d, shape, "llvm", j))
        register_task(Task("maxunpooling1d", "maxunpooling1d", maxunpooling1d, shape, "cuda", j))

for shape in maxunpooling2d_shape:
    for j in range(4):
        register_task(Task("maxunpooling2d", "maxunpooling2d", maxunpooling2d, shape, "llvm", j))
        register_task(Task("maxunpooling2d", "maxunpooling2d", maxunpooling2d, shape, "cuda", j))

for shape in PixelCNN_shape:
    # batch, H, W, in_C, out_C, KH, KW, mask_type, bias, dilation, stride, padding
    for j in range(4):
        register_task(Task("pixelcnn", "pixelcnn", pixelcnn, shape, "llvm", j))
        register_task(Task("pixelcnn", "pixelcnn", pixelcnn, shape, "cuda", j))

for shape in gated_pixelcnn_shape:
    for j in range(4):
        register_task(Task("gatedpixelcnn", "gatedpixelcnn", gatedpixelcnn, shape, "llvm", j))
        register_task(Task("gatedpixelcnn", "gatedpixelcnn", gatedpixelcnn, shape, "cuda", j))

for shape in shift_conv2d_shape:
    for j in range(4):
        register_task(Task("shift_conv2d", "shift_conv2d", shiftconv2d, shape, "llvm", j))
        register_task(Task("shift_conv2d", "shift_conv2d", shiftconv2d, shape, "cuda", j))
