# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
This file is modified from 
https://github.com/dmlc/tvm/blob/master/nnvm/python/nnvm/testing/mobilenet.py

last modify: 2019-09-21
editor: size zheng

This is a mobilenet-v1 benchmark implemented in NNVM
Notice that this implementation uses no optimization
so the performance should be poor.
"""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import time
import tvm
import nnvm
from nnvm import symbol as sym
import nnvm.testing.init as init
from tvm.contrib import graph_runtime
import numpy as np


def conv_block(data, name, channels,
               kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
               epsilon=1e-5):
    """Helper function to construct conv-bn-relu"""
    # convolution + bn + relu
    conv = sym.conv2d(data=data, channels=channels,
                      kernel_size=kernel_size, strides=strides,
                      padding=padding, use_bias=False,
                      layout="NCHW", name=name + "_conv")
    bn = sym.batch_norm(data=conv, epsilon=epsilon, name=name + "_bn")
    act = sym.relu(data=bn, name=name + "_relu")
    return act

def separable_conv_block(data, name, depthwise_channels,
                         pointwise_channels, kernel_size=(3, 3),
                         downsample=False, padding=(1, 1),
                         epsilon=1e-5):
    """Helper function to get a separable conv block"""
    if downsample:
        strides = (2, 2)
    else:
        strides = (1, 1)
    # depthwise convolution + bn + relu
    conv1 = sym.conv2d(data=data, channels=depthwise_channels,
                       groups=depthwise_channels, kernel_size=kernel_size, strides=strides,
                       padding=padding, use_bias=False, layout="NCHW",
                       name=name + "_depthwise_conv1")
    bn1 = sym.batch_norm(data=conv1, epsilon=epsilon, name=name + "_bn1")
    act1 = sym.relu(data=bn1, name=name + "_relu1")
    # pointwise convolution + bn + relu
    conv2 = sym.conv2d(data=act1, channels=pointwise_channels, kernel_size=(1, 1), strides=(1, 1),
                       padding=(0, 0), use_bias=False, layout="NCHW", name=name + "_conv2")
    bn2 = sym.batch_norm(data=conv2, epsilon=epsilon, name=name + "_bn2")
    act2 = sym.relu(data=bn2, name=name + "_relu2")
    return act2

def mobile_net(num_classes=1000, alpha=1.0, is_shallow=False):
    """Function to construct a MobileNet"""
    data = sym.Variable("data")
    body = conv_block(data, "conv_block_1", int(32*alpha), strides=(2, 2))
    body = separable_conv_block(body, "separable_conv_block_1",
                                int(32*alpha), int(64*alpha))
    body = separable_conv_block(body, "separable_conv_block_2",
                                int(64*alpha), int(128*alpha), downsample=True)
    body = separable_conv_block(body, "separable_conv_block_3",
                                int(128*alpha), int(128*alpha))
    body = separable_conv_block(body, "separable_conv_block_4",
                                int(128*alpha), int(256*alpha), downsample=True)
    body = separable_conv_block(body, "separable_conv_block_5",
                                int(256*alpha), int(256*alpha))
    body = separable_conv_block(body, "separable_conv_block_6",
                                int(256*alpha), int(512*alpha), downsample=True)
    if is_shallow:
        body = separable_conv_block(body, "separable_conv_block_7",
                                    int(512*alpha), int(1024*alpha), downsample=True)
        body = separable_conv_block(body, "separable_conv_block_8",
                                    int(1024*alpha), int(1024*alpha))
    else:
        for i in range(7, 12):
            body = separable_conv_block(body, "separable_conv_block_%d" % i,
                                        int(512*alpha), int(512*alpha))
        body = separable_conv_block(body, "separable_conv_block_12",
                                    int(512*alpha), int(1024*alpha), downsample=True)
        body = separable_conv_block(body, "separable_conv_block_13",
                                    int(1024*alpha), int(1024*alpha))
    pool = sym.global_avg_pool2d(data=body, name="pool")
    flatten = sym.flatten(data=pool, name="flatten")
    fc = sym.dense(data=flatten, units=num_classes, use_bias=False, name="fc")
    softmax = sym.softmax(data=fc, name="softmax")
    return softmax


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
    output = mobile_net()
    data_shape = (128, 3, 224, 224)
    trials = 400

    compute_graph = nnvm.graph.create(output)
    ctx = tvm.context("cuda", 0)
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
    
    # time_evaluator = module.module.time_evaluator("run", ctx, number=30, repeat=10)

    # time_cost = time_evaluator().mean * 1e3

    # print("time_cost=", time_cost, "ms")
    beg = time.time()
    for i in range(trials):
        module.run(data=input_data)
        output = module.get_output(0, None)
    end = time.time()
    print("end-to-end time cost=", (end - beg) * 1e3 / trials, "ms")



    
