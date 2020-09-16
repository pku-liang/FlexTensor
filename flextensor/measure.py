import tvm
import os
import numpy as np
import multiprocessing as mp
from flextensor.utils import RpcInfo, to_tuple


def verify_code(mod, target, dev_id):
    if target == "cuda":
        ctx = tvm.nd.context(target, dev_id)  # just use device 0
        if not ctx.exist:
            # print("Fail to get device %s devid=%d"%(target, dev_id))
            return False
        max_dims = ctx.max_thread_dimensions
        check_gpu = {
            "max_shared_memory_per_block": ctx.max_shared_memory_per_block,
            "max_threads_per_block": ctx.max_threads_per_block,
            "max_thread_x": max_dims[0],
            "max_thread_y": max_dims[1],
            "max_thread_z": max_dims[2]
        }
        valid = tvm.tir.ir_pass.VerifyGPUCode(mod, check_gpu)
        return valid
    else:
        # no barrier for other targets
        return True


def _evaluate(s, bufs, target, dev_id, number=1, q=None):
    ctx = tvm.context(target, dev_id)
    tvm_arys = []
    for arg in bufs:
        shape = to_tuple(arg.shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(arg.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    func, evaluator = None, None
    try:
        func = tvm.build(s, bufs, target)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
        time_cost = evaluator(*tvm_arys).mean * 1e3
        if q:
            q.put(time_cost)
        return time_cost
    except Exception as e:
        for item in tvm_arys:
            del item
        if func is not None:
            del func
        if evaluator is not None:
            del evaluator
        raise e


def servant_routine(func, que, *args, **kwargs):
    try:
        ret = func(*args, **kwargs)
    except Exception as e:
        ret = e
    que.put(ret)


def master_routine(timeout, func, *args, **kwargs):
    que = mp.Queue()
    servant = mp.Process(target=servant_routine, args=(func, que, *args), kwargs=kwargs)
    servant.start()
    try:
        ret = que.get(block=True, timeout=timeout)
    except Exception as e:
        ret = e
    servant.terminate()
    servant.join()
    return ret


def build_and_eval(lib, s, bufs, target, dev_id, rpc_info: RpcInfo = None, number=1):
    if rpc_info is not None:
        target_host = rpc_info.target_host
        fcompile = rpc_info.fcompile
        use_rpc = rpc_info.use_rpc
    else:
        target_host, fcompile, use_rpc = None, None, None

    # mod = tvm.lower(s, bufs, simple_mode=True)
    func = tvm.build(s, bufs, target=target, target_host=target_host)

    tvm_arys = []
    try:
        func.export_library(lib, fcompile)
        remote = rpc_info.get_remote()
        ctx = (remote if remote else tvm).context(target, dev_id)
        for buf in bufs:
            shape = to_tuple(buf.shape)
            tmp = np.random.uniform(0, 1, size=shape).astype(buf.dtype)
            tmp = tvm.nd.array(tmp, ctx)
            tvm_arys.append(tmp)

        if use_rpc:
            remote.upload(lib)
            func = remote.load_module(os.path.split(lib)[-1])
        else:
            func = tvm.runtime.module.load_module(lib)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
        time_cost = evaluator(*tvm_arys).mean * 1e3
    finally:
        while len(tvm_arys) > 0:
            del tvm_arys[-1]

    return time_cost
