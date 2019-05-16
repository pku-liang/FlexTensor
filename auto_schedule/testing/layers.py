"""
Base class for layers

====================================
**Author**: `Size Zheng`
"""
import tvm 
from auto_schedule.utils import assert_print
from auto_schedule.testing.ops import conv2d_nchw


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
        if isinstance(kernel_size, (int, tvm.expr.IntImm)):
            kernel_size = (kernel_size, kernel_size)
        assert_print(isinstance(kernel_size, tuple) and len(kernel_size) == 2)
        self.weight = tvm.placeholder((out_channel, in_channel, *kernel_size), dtype="float32")
        self.params["weight"] = self.weight
        if bias:
            self.bias = tvm.placeholder((out_channel,), dtype="float32")
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


class YoloConvLayer24(Conv2dLayer):
    """YOLO convolution Layer 1

    configuration:
    | name  | batch | channel |  H  |  W  | out_channel | k_h | k_w | padding | stride | 
    | conv1 |   x   |  1024   |  7  |  7  |     1024    |  3  |  3  |    1    |    1   |
    """
    def __init__(self):
        super(YoloConvLayer24, self).__init__(1024, 1024, 3, False, 1, 1, 1, 1)

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
        super(SqueezeNetFire8, self).__init__(384, 64, 27, True, 0, 1, 1, 1)

    @staticmethod
    def get_intput_shape():
        return (384, 27, 27)