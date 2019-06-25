import os
import tvm 
import topi
import numpy as np
from tvm import rpc


def to_int(expr):
    res = int(expr)
    return res


def to_tuple(expr_tuple):
    return tuple([to_int(x) for x in expr_tuple])



def gemm(data, weight, bias=None, out_dtype=None):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    matmul = tvm.compute((batch, out_dim), \
                         lambda i, j: tvm.sum(data[i, k].astype(out_dtype) * \
                                              weight[j, k].astype(out_dtype), axis=k), \
                         name='T_dense', tag='dense')
    if bias is not None:
        matmul = tvm.compute((batch, out_dim), \
                             lambda i, j: matmul[i, j] + bias[j].astype(out_dtype), \
                             tag=topi.tag.BROADCAST)
    return matmul


def schedule_dense(cfg, outs):
    """Schedule for dense operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    # pylint: disable=unused-argument
    # target = tvm.target.current_target()
    # if target.target_name == "cuda" and "cublas" in target.libs:
    #     return generic.schedule_extern(outs)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(Dense):
        num_thread = 64
        k = Dense.op.reduce_axis[0]
        ko, kf = s[Dense].split(k, factor=num_thread)
        DenseF = s.rfactor(Dense, kf)

        if Dense.op in s.outputs:
            Out = Dense
        else:
            Out = outs[0].op.output(0)
            s[Dense].compute_at(s[Out], s[Out].op.axis[1])
        s[Out].bind(s[Out].op.axis[0], tvm.thread_axis("blockIdx.y"))
        s[Out].bind(s[Out].op.axis[1], tvm.thread_axis("blockIdx.x"))

        tx = s[Dense].op.reduce_axis[0]
        thread_x = tvm.thread_axis("threadIdx.x")
        s[Dense].bind(tx, thread_x)
        s[DenseF].compute_at(s[Dense], tx)
        s[Dense].set_store_predicate(thread_x.var.equal(0))
        s[Out].set_store_predicate(thread_x.var.equal(0))

    scheduled_ops = []

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if topi.tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule dense
        elif OP.tag == 'dense':
            Dense = OP.output(0)
            _schedule(Dense)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


LOCAL_RPC = False
LIB_DIR = "."


def evaluate(name, s, bufs, target, dev_id, number, rpc_info=None):
    if rpc_info is not None:
        host = rpc_info.host
        port = rpc_info.port
    else:
        # local
        host = "0.0.0.0"
        port = 9090     # default port
    if host == "0.0.0.0":
        if LOCAL_RPC:
            use_rpc = True
        else:
            use_rpc = False
    else:
        use_rpc = True
    if use_rpc:
        remote = rpc.connect(host, port)
        ctx = remote.context(target, dev_id)
    else:
        ctx = tvm.context(target, dev_id)
    tvm_arys = []
    for buf in bufs:
        shape = to_tuple(buf.shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(buf.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    try:
        func_file = "{}.tar".format(name)
        if rpc_info is not None and rpc_info.target_host is not None:
            func = tvm.build(s, bufs, target=target, target_host=rpc_info.target_host)
        else:
            func = tvm.build(s, bufs, target=target)
        if use_rpc:
            func.export_library(os.path.join(LIB_DIR, func_file))
            remote.upload(os.path.join(LIB_DIR, func_file))
            func = remote.load_module(func_file)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
        time_cost = evaluator(*tvm_arys).mean * 1e3
    except Exception as e:
        # print(e)
        print("Bad schedule")
        time_cost = float("inf")
    finally:
        while len(tvm_arys) > 0:
            del tvm_arys[-1]
        if os.path.exists(os.path.join(LIB_DIR, func_file)):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
        elif os.path.exists(os.path.join(LIB_DIR, func_file + ".so")):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
    return time_cost


gemm_shapes = [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048)
]


if __name__ == "__main__":
    shapes = gemm_shapes
    names = ["gemm_" + str(i) + "_" + str(shapes[i]) for i in range(len(shapes))]
    for name, shape in zip(names, shapes):
        print("%s begins..." % name)
        N, K, M = shape
        A = tvm.placeholder([N, K])
        B = tvm.placeholder([M, K])
        Output = gemm(A, B)
        bufs = [A, B, Output]

        best_cost = float("inf")
        best_params = None
        s = schedule_dense(None, Output)
        cost = evaluate(name, s, bufs, "cuda", 0, 10, None)
        print("%s use %fms" % (name, cost))