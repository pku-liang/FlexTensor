from __future__ import absolute_import as _abs
import tvm
import nnvm
from nnvm import symbol as sym
import nnvm.testing.init as init
from tvm.contrib import graph_runtime
import numpy as np

"""
This implementation of MobileNet V2 is inspired
from https://github.com/tonylins/pytorch-mobilenet-v2

last modify: 2019-09-21
author: size zheng
"""

"""
This is a mobilenet-v2 benchmark implemented in NNVM
Notice that this implementation uses no optimization
so the performance should be poor.
"""

def generate_parameters(obj):
    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if isinstance(value, sym.Symbol):
                yield value
            elif isinstance(value, Layer):
                for ret in generate_parameters(value):
                    yield ret
            elif isinstance(value, (list, tuple)):
                for ele in value:
                    for ret in generate_parameters(ele):
                        yield ret
            elif isinstance(value, dict):
                for k, v in value.items():
                    for ret in generate_parameters(v):
                        yield ret


class Layer(object):
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        return generate_parameters(self)


def check_array(data, whole_type, inner_type, length):
    assert isinstance(length, int)
    assert isinstance(data, whole_type)
    assert len(data) == length
    for ele in data:
        assert isinstance(ele, inner_type)

    
def compose(f, *args):
    if len(args) == 0:
        return f 
    return compose(lambda *a: args[0](f(*a)), *args[1:])


def identity(inputs):
    return inputs


class Sequential(Layer):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._call_list = args

    def forward(self, inputs):
        # return compose(identity, *self._call_list)(*inputs)
        if len(self._call_list) == 0:
            return inputs 
        else:
            ret = self._call_list[0](inputs)
            for f in self._call_list[1:]:
                ret = f(ret)
            return ret


class Conv2d(Layer):
    def __init__(self, name, in_channel, out_channel, kernel_size, strides=1, padding=1, dilation=1, group=1, use_bias=False):
        if isinstance(strides, int):
            strides = (strides, strides) 
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        assert isinstance(group, int) 
        assert in_channel % group == 0 and out_channel % group == 0
        check_array(strides, tuple, int, 2)
        check_array(kernel_size, tuple, int, 2)
        check_array(padding, tuple, int, 2)
        check_array(dilation, tuple, int, 2)

        self.name = name
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._kernel_size = kernel_size
        self._padding = padding 
        self._strides = strides
        self._dilation = dilation 
        self._group = group 
        self._use_bias = use_bias 
        
        if group == in_channel:
            self.weight = sym.Variable("%s_weight" % name, shape=[in_channel, out_channel // in_channel, *kernel_size])
        else:
            self.weight = sym.Variable("%s_weight" % name, shape=[out_channel, in_channel//group, *kernel_size])
        if use_bias:
            self.bias = sym.Variable("%s_bias" % name, shape=[out_channel])

    def forward(self, inputs):
        if self._use_bias:
            return sym.conv2d(data=inputs, weight=self.weight, bias=self.bias, channels=self._out_channel, 
                    kernel_size=self._kernel_size, padding=self._padding, strides=self._strides, 
                    dilation=self._dilation, groups=self._group, use_bias=self._use_bias)
        else:
            return sym.conv2d(data=inputs, weight=self.weight, channels=self._out_channel, 
                    kernel_size=self._kernel_size, padding=self._padding, strides=self._strides, 
                    dilation=self._dilation, groups=self._group, use_bias=self._use_bias)


class BatchNorm(Layer):
    def __init__(self):
        super(BatchNorm, self).__init__()

    def forward(self, inputs):
        return sym.batch_norm(inputs)


class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, inputs):
        return sym.relu(inputs)


class Dropout(Layer):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self._rate = rate

    def forward(self, inputs):
        return sym.dropout(data=inputs, rate=self._rate)


class Dense(Layer):
    def __init__(self, name, in_dim, units, use_bias=False):
        super(Dense, self).__init__()
        self.name = name
        self._units = units
        self.weight = sym.Variable("%s_weight" % name, shape=[units, in_dim])
        if use_bias:
            self.bias = sym.Variable("%s_bias" % name, shape=[units])
        self._use_bias = use_bias

    def forward(self, inputs):
        if self._use_bias:
            return sym.dense(data=inputs, weight=self.weight, bias=self.bias, units=self._units)
        else:
            return sym.dense(data=inputs, weight=self.weight, units=self._units)


def conv_bn(name, in_channel, out_channel, kernel_size, strides=1, padding=1, dilation=1, group=1, use_bias=False):
    return Sequential(
        Conv2d(name, in_channel, out_channel, kernel_size, strides, padding, dilation, group, use_bias),
        BatchNorm(),
        ReLU()
    )


def conv_3x3_bn(name, in_channel, out_channel, strides):
    return conv_bn(name, in_channel, out_channel, 3, strides)


def conv_1x1_bn(name, in_channel, out_channel):
    return conv_bn(name, in_channel, out_channel, 1, padding=0)


class InvertedResidual(Layer):
    def __init__(self, name, in_channel, out_channel, strides, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.strides = strides 
        assert strides in [1, 2]

        hidden_dim = round(in_channel * expand_ratio)
        self.use_res_connect = self.strides == 1 and in_channel == out_channel

        if expand_ratio == 1:
            self.conv = Sequential(
                # depthwise
                Conv2d("%s_depthwise_1" % name, in_channel, hidden_dim, 3, strides=strides, padding=1, group=in_channel),
                BatchNorm(),
                ReLU(),
                # pointwise
                Conv2d("%s_pointwise_1" % name, hidden_dim, out_channel, 1, strides=1, padding=0),
                BatchNorm()
            )
        else:
            self.conv = Sequential(
                # pointwise
                Conv2d("%s_pointwise_1" % name, in_channel, hidden_dim, 1, strides=1, padding=0),
                BatchNorm(),
                ReLU(),
                # depthwise
                Conv2d("%s_depthwise_1" % name, hidden_dim, hidden_dim, 3, strides=strides, padding=1, group=hidden_dim),
                BatchNorm(),
                ReLU(),
                # pointwise
                Conv2d("%s_pointwise_2" % name, hidden_dim, out_channel, 1, strides=1, padding=0),
                BatchNorm()
            )

    def forward(self, inputs):
        if self.use_res_connect:
            return inputs + self.conv(inputs)
        else:
            return self.conv(inputs)


class MobileNetV2(Layer):
    def __init__(self, name, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # expand_ratio, c, n, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_3x3_bn("%s_first_conv3x3_bn" % name, 3, input_channel, 2)]

        for count, (t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block("%s_residual_%d_%d" % (name, count+1, i+1), 
                                    input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block("%s_residual_%d_%d" % (name, count+1, i+1), 
                                    input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
                
        self.features.append(conv_1x1_bn("%s_last_conv1x1_bn" % name, input_channel, self.last_channel))
        self.features = Sequential(*self.features)

        self.classifier = Sequential(
            Dropout(0.2),
            Dense("%s_dense" % name, self.last_channel, n_class)
        )

    def forward(self, inputs):
        y = self.features(inputs)
        y = sym.mean(y, axis=[2, 3])
        y = self.classifier(y)
        return y 


def generate_random_parameters(compute_graph, input_name, input_shape, with_input=False, 
                            context=tvm.cpu(0), seed=0, dtype="float32"):
    input_shapes, _ = nnvm.compiler.graph_util.infer_shape(compute_graph, data=input_shape)
    params = {}
    shape_dict = dict(zip(compute_graph.index.input_names, input_shapes))
    np.random.seed(seed)
    for name, shape in shape_dict.items():
        if name == input_name and not with_input:
            continue
        else:
            init_value = np.random.uniform(-1, 1, shape).astype(dtype)
            params[name] = tvm.nd.array(init_value, ctx=context)
    return params


if __name__ == "__main__":
    net = MobileNetV2("mobilenet-v2")
    data_shape = (1, 3, 224, 224)

    inputs = sym.Variable("data", shape=data_shape)
    output = net(inputs)
    
    compute_graph = nnvm.graph.create(output)
    ctx = tvm.context("cuda", 0)
    params = generate_random_parameters(compute_graph, "data", data_shape, with_input=True, context=ctx)
    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target="cuda", 
                                        shape={"data": data_shape}, params=params)
    print(deploy_graph.ir())
    module = graph_runtime.create(deploy_graph, lib, ctx)
    
    time_evaluator = module.module.time_evaluator("run", ctx, number=20, repeat=10)

    time_cost = time_evaluator().mean * 1e3

    print("time_cost=", time_cost, "ms")