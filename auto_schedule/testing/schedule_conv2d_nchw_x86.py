"""
High performance schedule for conv2d_nchw
Target x86 CPU

====================================
**Author**: `Size Zheng`
"""
import tvm 
from auto_schedule.testing.ops import conv2d_nchw
from auto_schedule.testing.layers import Layer
from auto_schedule.utils import assert_print
from auto_schedule.measure import _evaluate


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


def schedule_yolo_conv_cuda(s, outputs, inputs, weight):
    # inline the padding operation
    padded = outputs.op.input_tensors[0]
    

    # create cache
    write_cache = s.cache_write(outputs, "local")
    read_share_weight = s.cache_read(weight, "shared", [write_cache])
    read_local_weight = s.cache_read(read_share_weight, "local", [write_cache])
    read_share_inputs = s.cache_read(padded, "shared", [write_cache])
    read_local_inputs = s.cache_read(read_share_inputs, "local", [write_cache])

    # tunable parameters
    b_factors = [1, 1, 1, 1]
    k_factors = [64, 2, 8, 1]
    p_factors = [1, 1, 7, 1]
    q_factors = [1, 1, 7, 1]
    rc_factors = [64, 1, 16]         # outer-->inner
    ry_factors = [1, 3, 1]
    rx_factors = [1, 1, 3]

    # prepare thread_axis
    bx = tvm.thread_axis("blockIdx.x")
    by = tvm.thread_axis("blockIdx.y")
    bz = tvm.thread_axis("blockIdx.z")
    vx = tvm.thread_axis("vthread")
    vy = tvm.thread_axis("vthread")
    vz = tvm.thread_axis("vthread")
    tx = tvm.thread_axis("threadIdx.x")
    ty = tvm.thread_axis("threadIdx.y")
    tz = tvm.thread_axis("threadIdx.z")

    # split the spatial axes
    b, k, p, q = s[outputs].op.axis
    bo, bi = s[outputs].split(b, nparts=b_factors[0])
    ko, ki = s[outputs].split(k, nparts=k_factors[0])
    po, pi = s[outputs].split(p, nparts=p_factors[0])
    qo, qi = s[outputs].split(q, nparts=q_factors[0])

    vbo, bi = s[outputs].split(bi, nparts=b_factors[1])
    vko, ki = s[outputs].split(ki, nparts=k_factors[1])
    vpo, pi = s[outputs].split(pi, nparts=p_factors[1])
    vqo, qi = s[outputs].split(qi, nparts=q_factors[1])

    tbo, bi = s[outputs].split(bi, nparts=b_factors[2])
    tko, ki = s[outputs].split(ki, nparts=k_factors[2])
    tpo, pi = s[outputs].split(pi, nparts=p_factors[2])
    tqo, qi = s[outputs].split(qi, nparts=q_factors[2])

    # reorder
    s[outputs].reorder(bo, ko, po, qo, vbo, vko, vpo, vqo, tbo, tko, tpo, tqo, bi, ki, pi, qi)

    # fuse
    outer = s[outputs].fuse(bo, ko, po, qo)
    middle = s[outputs].fuse(vbo, vko, vpo, vqo)
    inner = s[outputs].fuse(tbo, tko, tpo, tqo)
    left = s[outputs].fuse(bi, ki, pi, qi)

    # bind
    s[outputs].bind(outer, bx)
    s[outputs].bind(middle, vx)
    s[outputs].bind(inner, tx)

    # compute at write cache
    s[write_cache].compute_at(s[outputs], inner)

    # split reduce axes
    wb, wk, wp, wq = s[write_cache].op.axis
    rc, ry, rx = s[write_cache].op.reduce_axis
    rco, rci = s[write_cache].split(rc, nparts=rc_factors[0])
    rcm, rci = s[write_cache].split(rci, nparts=rc_factors[1])
    rxo, rxi = s[write_cache].split(rx, nparts=rx_factors[0])
    rxm, rxi = s[write_cache].split(rxi, nparts=rx_factors[1])
    ryo, ryi = s[write_cache].split(ry, nparts=ry_factors[0])
    rym, ryi = s[write_cache].split(ryi, nparts=ry_factors[1])

    # reorder
    s[write_cache].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, wb, wk, wp, wq)

    # compute at read cache
    s[read_share_weight].compute_at(s[write_cache], rcm)
    s[read_local_weight].compute_at(s[write_cache], rxi)
    s[read_share_inputs].compute_at(s[write_cache], rcm)
    s[read_local_inputs].compute_at(s[write_cache], rxi)

    # cooperative fetching
    for cache in [read_share_inputs, read_share_weight]:
        cb, ck, ch, cw = s[cache].op.axis
        fused = s[cache].fuse(cb, ck, ch, cw)
        tx_extent = b_factors[2] * k_factors[2] * p_factors[2] * q_factors[2]
        bindx, fused = s[cache].split(fused, nparts=tx_extent)
        s[cache].bind(bindx, tx)
    
    s[outputs].pragma(outer, 'auto_unroll_max_step', 1024)
    s[outputs].pragma(outer, 'unroll_explicit', 1)

    s[padded].compute_inline()


def try_yolo_conv(batch_size=1):
    # get the compute
    yolo_conv24 = YoloConvLayer24()
    input_shape = yolo_conv24.get_intput_shape()
    inputs = tvm.placeholder((batch_size, *input_shape), dtype="float32")
    weight = yolo_conv24.get_weight()
    outputs = yolo_conv24(inputs)
    
    s = tvm.create_schedule(outputs.op)
    schedule_yolo_conv_cuda(s, outputs, inputs, weight)

    arg_bufs = [inputs, weight, outputs]
    print(tvm.lower(s, arg_bufs, simple_mode=True))
    time_cost = _evaluate(s, arg_bufs, "cuda", 0, 100)
    print("Yolo conv1 use", time_cost, "ms")


if __name__ == "__main__":
    try_yolo_conv(batch_size=1)