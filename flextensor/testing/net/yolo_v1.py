from __future__ import absolute_import

import time
import tvm
import nnvm 
from nnvm import symbol as sym 
import nnvm.testing.init as init 
from tvm.contrib import graph_runtime
import numpy as np


def conv2d_block(data, name, channels, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)):
    conv2d = sym.conv2d(
        data=data, 
        channels=channels, 
        kernel_size=kernel_size, 
        strides=strides, 
        padding=padding, 
        use_bias=False, 
        layout="NCHW", 
        name=name + "_conv2d"
    )
    # act = sym.relu(data=conv2d, name=name + "_relu")
    return conv2d


def yolo(num_classes=1470):
    data = sym.Variable("data")
    body = conv2d_block(data, "conv1", 64, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3))
    body = sym.max_pool2d(data=body, pool_size=(2, 2), strides=(2, 2), name="pool1")
    body = conv2d_block(body, "conv2", 192, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = sym.max_pool2d(data=body, pool_size=(2, 2), strides=(2, 2), name="pool2")

    body = conv2d_block(body, "conv3", 128, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
    body = conv2d_block(body, "conv4", 256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = conv2d_block(body, "conv5", 256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
    body = conv2d_block(body, "conv6", 512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = sym.max_pool2d(data=body, pool_size=(2, 2), strides=(2, 2), name="pool3")

    body = conv2d_block(body, "conv7", 256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
    body = conv2d_block(body, "conv8", 512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = conv2d_block(body, "conv9", 256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
    body = conv2d_block(body, "conv10", 512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = conv2d_block(body, "conv11", 256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
    body = conv2d_block(body, "conv12", 512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = conv2d_block(body, "conv13", 256, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
    body = conv2d_block(body, "conv14", 512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = conv2d_block(body, "conv15", 512, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
    body = conv2d_block(body, "conv16", 1024, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = sym.max_pool2d(data=body, pool_size=(2, 2), strides=(2, 2), name="pool4")

    body = conv2d_block(body, "conv17", 512, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
    body = conv2d_block(body, "conv18", 1024, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = conv2d_block(body, "conv19", 512, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))
    body = conv2d_block(body, "conv20", 1024, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = conv2d_block(body, "conv21", 1024, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = conv2d_block(body, "conv22", 1024, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1))
    body = conv2d_block(body, "conv23", 1024, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
    body = conv2d_block(body, "conv24", 1024, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))

    flatten = sym.flatten(data=body, name="flatten")
    fc = sym.dense(data=flatten, units=4096, use_bias=False, name="fc1")
    act = sym.relu(data=fc, name="relu1")
    fc = sym.dense(data=act, units=num_classes, use_bias=False, name="fc2")

    return fc 


def single_net():
    data = sym.Variable("data")
    body = conv2d_block(data, "conv1", 64, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3))
    return body


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


def get_workload(batch_size=1, image_shape=(3, 448, 448), dtype="float32"):
    """Get benchmark workload for a DCGAN generator

    Parameters
    ----------
    batch_size : int
        The batch size used in the model
    image_shape: tuple
    dtype : str, optional
        The data type

    Returns
    -------
    net : nnvm.symbol
        The computational graph
    params : dict of str to NDArray
        The parameters.
    """
    net = yolo()
    return nnvm.testing.utils.create_workload(net, batch_size, image_shape, dtype)


if __name__ == "__main__":    
    output = yolo()
    data_shape = (1, 3, 448, 448)
    trials = 50

    compute_graph = nnvm.graph.create(output)
    ctx = tvm.device("cuda", 0)
    params = generate_random_parameters(compute_graph, "data", data_shape, with_input=True, context=ctx)
    input_data = params["data"]
    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target="cuda", 
                                        shape={"data": data_shape}, params=params)
    # print(deploy_graph.ir())
    module = graph_runtime.create(deploy_graph, lib, ctx)

    # warm-up
    module.run(data=input_data)
    output = module.get_output(0, None)
    # print(output.asnumpy())
    
    time_evaluator = module.module.time_evaluator("run", ctx, number=trials, repeat=10)

    time_cost = time_evaluator().mean * 1e3

    print("time_cost=", time_cost, "ms")

    # beg = time.time()
    # for i in range(trials):
    #     module.run(data=input_data)
    #     output = module.get_output(0, None)
    # end = time.time()
    # print("end-to-end time cost=", (end - beg) * 1e3 / trials, "ms")