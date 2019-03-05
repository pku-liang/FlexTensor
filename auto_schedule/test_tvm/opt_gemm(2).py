import logging
import sys
import numpy as np

import tvm
import topi
from topi.testing import conv2d_nchw_python

from tvm import autotvm
from xgboost import plot_tree
import time
total_=64*15*15
batch, M, N, K = 36, total_, 192, 48

def make_kernel(total_, N, K, nparts_, blockdimx, griddimx, griddimy, step):
    blockdimy=int(128/blockdimx)
    A = tvm.placeholder((int(total_/nparts_), 36, nparts_, N), name='A')
    B = tvm.placeholder((36, K, N), name='B')
    rk = tvm.reduce_axis((0, N), name='rk')

    # Pad input
    C = tvm.compute(
        (int(total_/nparts_), 36, nparts_, K),
        lambda pp, bb, mm, nn: tvm.sum(
            A[pp, bb, mm, rk] * B[bb, nn, rk],
            axis=[rk]),
        name='C')
    s = tvm.create_schedule(C.op)
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CL = s.cache_write(C, "local")
    ##### space definition begin #####
    pa, ba, ma, na = s[C].op.axis   #B=36, M=256, N=1024
    # block_x = tvm.thread_axis("blockIdx.x")
    # block_y = tvm.thread_axis("blockIdx.y")
    # block_z = tvm.thread_axis("blockIdx.z")
    # thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    # thread_y = tvm.thread_axis((0, 16), "threadIdx.y")
    # thread_xz = tvm.thread_axis((0, 4), "vthread", name="vx")
    # thread_yz = tvm.thread_axis((0, 16), "vthread", name="vy")
    ba = s[C].fuse(pa,ba)
    by, ma=s[C].split(ma,nparts=griddimy)
    vy, ty=s[C].split(ma,factor=blockdimy)
    bx, na=s[C].split(na,nparts=griddimx)
    vx, tx=s[C].split(na,factor=blockdimx)
    s[C].reorder(ba,by,bx,vy,vx,ty,tx)
    s[C].bind(ba, tvm.thread_axis("blockIdx.z"))
    s[C].bind(by, tvm.thread_axis("blockIdx.y"))
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(vx, tvm.thread_axis("vthread"))
    s[C].bind(vy, tvm.thread_axis("vthread"))
    s[C].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[CL].compute_at(s[C], tx) # threadidx.x level calc BL
    rk = s[CL].op.reduce_axis[0]
    rko,rki=s[CL].split(rk,factor=step)
    s[AA].compute_at(s[CL], rko) #BL rx iteration calc AA and WW
    s[BB].compute_at(s[CL], rko)
    s[AL].compute_at(s[CL], rki)
    s[BL].compute_at(s[CL], rki)

    pp, bb, mm, kk = s[AA].op.axis

    # mk = s[AA].fuse(mm,kk)
    # mk,tx = s[AA].split(mk,factor=blockdimx)
    # mk,ty = s[AA].split(mk,factor=blockdimy)
    tx, kk = s[AA].split(kk, nparts=blockdimx)
    ty, mm = s[AA].split(mm, nparts=blockdimy)
    
    s[AA].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[AA].bind(tx, tvm.thread_axis("threadIdx.x"))

    bb, mm, kk = s[BB].op.axis
    # mk = s[BB].fuse(mm,kk)
    # mk, tx = s[BB].split(mk,factor=blockdimx)
    # mk, ty = s[BB].split(mk,factor=blockdimy)
    tx, kk = s[BB].split(kk, nparts=blockdimx)
    ty, mm = s[BB].split(mm, nparts=blockdimy)
    
    s[BB].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[BB].bind(tx, tvm.thread_axis("threadIdx.x"))

    func=tvm.build(s,[A,B,C], 'cuda')
    ctx = tvm.gpu(0)
    a_np = np.random.uniform(size=(int(total_/nparts_), 36, nparts_, N)).astype(A.dtype)
    b_np = np.random.uniform(size=(36, K, N)).astype(B.dtype)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((int(total_/nparts_), 36, nparts_, K), dtype=C.dtype), ctx)
    print(tvm.lower(s, [A, B, C],simple_mode=True))
    print(func.imported_modules[0].get_source())
    func(a, b, c)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=1,repeat=10)
    cost_time=evaluator(a, b, c).mean * 1e3
    print('%d %d %d %d %d: %f ms' % (nparts_,blockdimx, griddimx, griddimy, step,cost_time))
    return cost_time

set_nparts=[32,64,128]      #nparts should be as small as possible, with ~10% tolerance of performance slowdown.
set_blockdimx=[4,8,16,32] #blockdimy=128/blockdimx
set_griddimx=[1,2,4]    # probably can change griddim
set_griddimy=[1]
set_step=[4,8,16]  #step=4,8 or 16

# set_nparts=[64]      #nparts should be as small as possible, with ~10% tolerance of performance slowdown.
# set_blockdimx=[16] #blockdimy=128/blockdimx
# set_griddimx=[1]    # probably can change griddim
# set_griddimy=[1]
# set_step=[16]  #step=4,8 or 16

configs=[
    (128*2*2,512,512),   #ResNet-Res5
    (128*2*2,512,1024),  #YOLO-conv5
]


for total, N, K in configs:
    best_time=1000
    best_param=[]
    for a1 in set_nparts:
        for a2 in set_blockdimx:
            for a3 in set_griddimx:
                for a4 in set_griddimy:
                    for a5 in set_step:
                        time=make_kernel(total, N, K, a1,a2,a3,a4,a5)
                        if time<best_time:
                            best_time=time
                            best_param=[a1,a2,a3,a4,a5]
    print(best_param)
    print(best_time)
