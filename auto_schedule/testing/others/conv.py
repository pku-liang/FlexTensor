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
.. _opt-conv-gpu:

How to optimize convolution on GPU
==================================
**Author**: `Haichen Shen <https://homes.cs.washington.edu/~haichen/>`_

In this tutorial, we will demonstrate how to write a high performance
convolution implementation in TVM. We use square size input tensors and filters
as an example, and assume the input to convolution has a large batch. In this
example, we use a different layout to store the data in order to achieve better
data locality. The buffer layout is HWCN, which stands for height, width,
channel, batch.

"""

################################################################
# Preparation and Algorithm
# -------------------------
#
# We use the fixed size for input tensors with 256 channels and 14 x 14
# dimensions. The batch size is 256. Convolution filters contain 512 filters
# of size 3 x 3.  We use stride size 1 and padding size 1 for the
# convolution. The following code defines the convolution algorithm in TVM.
#

import numpy as np
import tvm

# The sizes of inputs and filters
N = 256
C = 256
K = 512
H= 14
kernel = 1
pad = 0
stride = 1

# Algorithm
A = tvm.placeholder((N, C, H, H), name='A')
W = tvm.placeholder((K, C), name='W')
# Pad input

# Create reduction variables
rc = tvm.reduce_axis((0, C), name='rc')

# Compute the convolution
B = tvm.compute(
    (N, K, H, H),
    lambda nn, ff, yy, xx: tvm.sum(
        A[nn, rc, yy, xx] * W[ff, rc],
        axis=[rc]),
    name='B')


###############################################################################
# Memory Hierarchy
# ----------------
#
# We first specify the memory hierarchy for buffers. The figure below shows the
# GPU memory hierarchy. One important difference from CPU memory hierarchy is
# that GPU provides a cache buffer called shared memory, which is managed by
# programmers. Thus how to maximize the data reuse in the shared memory is
# critical to achieve high performance in GPU kernels.
#
# .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/gpu_memory_hierarchy.png
#      :align: center
#      :height: 319px
#      :width: 271px
#
# In this example, we load both Apad and W into buffer AA and WW, which are
# stored in the shared memory. These bufferes will be later shared by all
# threads within the same thread block to compute the convolution. Each thread
# then loads its own part from shared buffer into their local registers, AL and
# WL. BL is a local cache of output B, which is also stored in the thread local
# registers.
#

# Designate the memory hierarchy

# blockdimx_s = [8,16]
# blockdimy_s = [8,16]
# threaddimy_s = [4,8,16]

# step_s = [8,16]
# gridx_h_s = [1,2]
# gridx_w_s = [1,2]

blockdimx_s = [16]
blockdimy_s = [8]
threaddimy_s = [8]

step_s = [16]
#threaddimx_s = [4,6,7,8,9,10,11,12,14]
# griddimx_s=[192, 256, 320, 392, 448, 512]

def make_schedule(blockdimx, blockdimy, threaddimy, step):
    s = tvm.create_schedule(B.op)
    AA = s.cache_read(A, 'shared', [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BL = s.cache_write(B, "local")

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, blockdimx), "threadIdx.x")
    thread_y = tvm.thread_axis((0, blockdimy), "threadIdx.y")
    thread_xz = tvm.thread_axis("vthread")
    thread_yz = tvm.thread_axis("vthread")


    ni, fi, hi, wi = s[B].op.axis

    hw = s[B].fuse(hi, wi)

    by, fi = s[B].split(fi, factor=threaddimy * blockdimy)

    # Bind the iteration variables to GPU thread indices
    s[B].bind(ni, block_z)
    s[B].bind(by, block_y)
    #s[B].bind(hwo, block_x)

    fi, ty = s[B].split(fi, factor=blockdimy)
    ni, tx = s[B].split(hw, factor=blockdimx)
    #s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

    s[B].bind(fi, thread_yz)
    s[B].bind(ni, thread_xz)
    s[B].bind(ty, thread_y)
    s[B].bind(tx, thread_x)


    # Schedule BL local write
    s[BL].compute_at(s[B], tx)
    ni, fi, yi, xi = s[BL].op.axis
    rc, = s[BL].op.reduce_axis
    rco, rci = s[BL].split(rc, factor=step)

    # Attach computation to iteration variables
    s[AA].compute_at(s[BL], rco)
    s[WW].compute_at(s[BL], rco)
    s[AL].compute_at(s[BL], rci)
    s[WL].compute_at(s[BL], rci)

    # Schedule for A's shared memory load
    yi, xi, ci, ni = s[AA].op.axis
    ty, ci = s[AA].split(ci, nparts=blockdimy)
    tx, ni = s[AA].split(ni, nparts=blockdimx)
    s[AA].reorder(ty, tx, yi, xi, ci, ni)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    #s[AA].vectorize(ni)  # vectorize memory load

    # Schedule for W's shared memory load
    ci, fi= s[WW].op.axis
    ty, ci = s[WW].split(ci, nparts=blockdimy)
    tx, fi = s[WW].split(fi, nparts=blockdimx)
    s[WW].reorder(ty, tx, ci, fi)
    s[WW].bind(ty, thread_y)
    s[WW].bind(tx, thread_x)
    #s[WW].vectorize(fi)  # vectorize memory load


    ###############################################################################
    # Generate CUDA Kernel
    # --------------------
    #
    # Finally we use TVM to generate and compile the CUDA kernel, and evaluate the
    # latency of convolution.
    #

    func = tvm.build(s, [A, W, B], 'cuda')
    ctx = tvm.gpu(0)
    a_np = np.random.uniform(size=(N, C, H, H)).astype(A.dtype)
    w_np = np.random.uniform(size=(K, C)).astype(W.dtype)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros((N, K, H, H), dtype=B.dtype), ctx)
    func(a, w, b)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
    #print(func.imported_modules[0].get_source())
    with open('kernel.h','w') as f:
        f.write(func.imported_modules[0].get_source())
    print('%d %d %d %d: %f ms' %  (blockdimx, blockdimy, threaddimy, step,evaluator(a, w, b).mean * 1e3))


for i in blockdimx_s:
    for j in blockdimy_s:
        for k in threaddimy_s:
            for l in step_s:
                #for m in griddimx_s:
                    make_schedule(i,j,k,l)