"""
Base class for layers
====================================
**Author**: `Size Zheng`
"""
import tvm 
from flextensor.utils import assert_print
from .ops import conv2d_nchw
from .ops import gemm_conv2d_nchw


class Layer(object):
    """Base class for layers
    """
    def __init__(self):
        self.params = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def parameters(self):
        return self.params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Conv2dLayer(Layer):
    """Convolution 2D Layer
    """
    def __init__(self, in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1):
        super(Conv2dLayer, self).__init__()
        if isinstance(kernel_size, (int, tvm.tir.IntImm)):
            kernel_size = (kernel_size, kernel_size)
        assert_print(isinstance(kernel_size, tuple) and len(kernel_size) == 2)
        self.weight = tvm.te.placeholder((out_channel, in_channel, *kernel_size), dtype="float32")
        self.params["weight"] = self.weight
        if bias:
            self.bias = tvm.te.placeholder((out_channel,), dtype="float32")
            self.params["bias"] = self.bias
        else:
            self.bias = None

        def forward_func(inputs):
            return conv2d_nchw(inputs, self.weight, self.bias, stride, padding, dilation, groups)
        self.forward_func = forward_func

    def get_weight(self):
        return self.params["weight"]
    
    def get_bias(self):
        if "bias" in self.params:
            return self.bias
        else:
            return None
    
    def forward(self, inputs):
        return self.forward_func(inputs)


class YoloConvLayer1(Conv2dLayer):
    """YOLO convolution Layer 1
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv1 |   x   |    3    | 448 | 448 |     64      |  7  |  7  |    3    |    2   |
    """
    def __init__(self):
        super(YoloConvLayer1, self).__init__(3, 64, 7, False, 3, 2, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (3, 448, 448)


class YoloConvLayer2(Conv2dLayer):
    """YOLO convolution Layer 2
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv2 |   x   |    64   | 112 | 112 |     192      |  3  |  3  |    1    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer2, self).__init__(64, 192, 3, False, 1, 1, 1, 1) 
        # in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1

    @staticmethod
    def get_intput_shape():
        return (64, 112, 112)


class YoloConvLayer3(Conv2dLayer):
    """YOLO convolution Layer 3
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv3 |   x   |   192   |  56 |  56 |    128      |  1  |  1  |    0    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer3, self).__init__(192, 128, 1, False, 0, 1, 1, 1) 
        # in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1

    @staticmethod
    def get_intput_shape():
        return (192, 56, 56)


class YoloConvLayer4(Conv2dLayer):
    """YOLO convolution Layer 4
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv4 |   x   |   128   |  56 |  56 |    256      |  3  |  3  |    1    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer4, self).__init__(128, 256, 3, False, 1, 1, 1, 1) 
        # in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1

    @staticmethod
    def get_intput_shape():
        return (128, 56, 56)


class YoloConvLayer5(Conv2dLayer):
    """YOLO convolution Layer 5
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv5 |   x   |   256   |  56 |  56 |    256      |  1  |  1  |    0    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer5, self).__init__(256, 256, 1, False, 0, 1, 1, 1) 
        # in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1

    @staticmethod
    def get_intput_shape():
        return (256, 56, 56)


class YoloConvLayer6(Conv2dLayer):
    """YOLO convolution Layer 6
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv1 |   x   |   256   |  56 |  56 |    512      |  3  |  3  |    1    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer6, self).__init__(256, 512, 3, False, 1, 1, 1, 1) 
        # in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1

    @staticmethod
    def get_intput_shape():
        return (256, 56, 56)


class YoloConvLayer7(Conv2dLayer):
    """YOLO convolution Layer 7
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv7 |   x   |   512   |  28 |  28 |    256      |  1  |  1  |    0    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer7, self).__init__(512, 256, 1, False, 0, 1, 1, 1) 
        # in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1

    @staticmethod
    def get_intput_shape():
        return (512, 28, 28)


class YoloConvLayer8(Conv2dLayer):
    """YOLO convolution Layer 8
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv8 |   x   |   256   |  28 |  28 |    512      |  3  |  3  |    1    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer8, self).__init__(256, 512, 3, False, 1, 1, 1, 1) 
        # in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1

    @staticmethod
    def get_intput_shape():
        return (256, 28, 28)


class YoloConvLayer9(Conv2dLayer):
    """YOLO convolution Layer 9
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv9 |   x   |   512   |  28 |  28 |    512      |  1  |  1  |    0    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer9, self).__init__(512, 512, 1, False, 0, 1, 1, 1) 
        # in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1

    @staticmethod
    def get_intput_shape():
        return (512, 28, 28)


class YoloConvLayer10(Conv2dLayer):
    """YOLO convolution Layer 10
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv10|   x   |   512   |  28 |  28 |    1024     |  3  |  3  |    1    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer10, self).__init__(512, 1024, 3, False, 1, 1, 1, 1) 
        # in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1

    @staticmethod
    def get_intput_shape():
        return (512, 28, 28)


class YoloConvLayer11(Conv2dLayer):
    """YOLO convolution Layer 11
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv11|   x   |   1024  |  14 |  14 |     512     |  1  |  1  |    0    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer11, self).__init__(1024, 512, 1, False, 0, 1, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (1024, 14, 14)


class YoloConvLayer12(Conv2dLayer):
    """YOLO convolution Layer 12
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv12|   x   |    512  |  14 |  14 |     1024    |  3  |  3  |    1    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer12, self).__init__(512, 1024, 3, False, 1, 1, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (512, 14, 14)


class YoloConvLayer13(Conv2dLayer):
    """YOLO convolution Layer 13
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv13|   x   |   1024  |  14 |  14 |     1024    |  3  |  3  |    1    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer13, self).__init__(1024, 1024, 3, False, 1, 1, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (1024, 14, 14)


class YoloConvLayer14(Conv2dLayer):
    """YOLO convolution Layer 14
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv14|   x   |   1024  |  14 |  14 |     1024    |  3  |  3  |    1    |    2   |
    """
    def __init__(self):
        super(YoloConvLayer14, self).__init__(1024, 1024, 3, False, 1, 2, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (1024, 14, 14)


class YoloConvLayer15(Conv2dLayer):
    """YOLO convolution Layer 15
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv15|   x   |  1024   |  7  |  7  |     1024    |  3  |  3  |    1    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer15, self).__init__(1024, 1024, 3, False, 1, 1, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (1024, 7, 7)


class SqueezeNetFire8(Conv2dLayer):
    """SqueezeNet convolution Layer 8
    Configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv8 |   x   |   384   | 27  | 27  |      64     |  1  |  1  |    0    |    1   |
    """
    def __init__(self):
        super(SqueezeNetFire8, self).__init__(384, 64, 1, True, 0, 1, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (384, 27, 27)

"""
    Gemm conv2d layers
"""
class GemmConv2dLayer(Layer):
    """Convolution 2D Layer
    """
    def __init__(self, in_channel, out_channel, kernel_size, bias=False, padding=0, stride=1, dilation=1, groups=1):
        super(GemmConv2dLayer, self).__init__()
        if isinstance(kernel_size, (int, tvm.tir.IntImm)):
            kernel_size = (kernel_size, kernel_size)
        assert_print(isinstance(kernel_size, tuple) and len(kernel_size) == 2)
        self.weight = tvm.te.placeholder((out_channel, in_channel, *kernel_size), dtype="float32", name='weights')
        self.weight_shape = (out_channel, in_channel, *kernel_size)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.params["weight"] = self.weight
        if bias:
            self.bias = tvm.te.placeholder((out_channel,), dtype="float32", name='bias')
            self.params["bias"] = self.bias
        else:
            self.bias = None

        def forward_func(inputs):
            return gemm_conv2d_nchw(inputs, self.weight, self.bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.forward_func = forward_func

    def get_weight(self):
        return self.params["weight"]
    
    def get_bias(self):
        if "bias" in self.params:
            return self.bias
        else:
            return None
    
    def forward(self, inputs):
        return self.forward_func(inputs)

class YoloGemmConvLayer1(GemmConv2dLayer):
    """YOLO convolution Layer 1
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv1 |   x   |    3    | 448 | 448 |     64      |  7  |  7  |    3    |    2   |
    """
    def __init__(self):
        super(YoloGemmConvLayer1, self).__init__(3, 64, 7, False, 3, 2, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (3, 448, 448)


class YoloGemmConvLayer17(GemmConv2dLayer):
    """YOLO convolution Layer 1
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv1 |   x   |   1024  |  14 |  14 |     512     |  1  |  1  |    0    |    1   |
    """
    def __init__(self):
        super(YoloGemmConvLayer17, self).__init__(1024, 512, 1, False, 0, 1, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (1024, 14, 14)


class YoloGemmConvLayer24(GemmConv2dLayer):
    """YOLO convolution Layer 1
    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv1 |   x   |  1024   |  7  |  7  |     1024    |  3  |  3  |    1    |    1   |
    """
    def __init__(self):
        super(YoloGemmConvLayer24, self).__init__(1024, 1024, 3, False, 1, 1, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (1024, 7, 7)


class SqueezeNetFire8Gemm(GemmConv2dLayer):
    """SqueezeNet convolution Layer 8
    Configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv8 |   x   |   384   | 27  | 27  |      64     |  1  |  1  |    0    |    1   |
    """
    def __init__(self):
        super(SqueezeNetFire8Gemm, self).__init__(384, 64, 1, True, 0, 1, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (384, 27, 27)