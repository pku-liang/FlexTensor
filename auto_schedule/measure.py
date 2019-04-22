import tvm
import signal
import psutil
import time
import numpy as np
import torch.multiprocessing as multi
from auto_schedule.utils import to_tuple


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """kill all child processes recursively"""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            return


def parallel_evaluate(s, bufs, target, dev_id, number=10, timeout=10.0):
    proc = []
    q = multi.Queue()
    for i in range(number):
        p = multi.Process(target=_evaluate, args=(s, bufs, target, dev_id, 1, q))
        p.start()
        proc.append(p)
    beg = time.time()
    while time.time() - beg < timeout:
        if any(p.is_alive() for p in proc):
            time.sleep(.1)
        else:
            break
    else:
        for p in proc:
            p.terminate()
            p.join()
    count = 0
    sum = 0
    while not q.empty():
        sum += q.get()
        count += 1
    while count < number:
        sum += timeout * 1e3
        count += 1
    return sum / count


def serial_evaluate(s, bufs, target, dev_id, number=10, timeout=10.0):
    que = multi.Queue()
    total_timeout = timeout * number

    def _evaluate_loop(q):
        ret = []
        for i in range(number):
            cost = _evaluate(s, bufs, target, dev_id)
            ret.append(cost)
        q.put(ret)

    p = multi.Process(target=_evaluate_loop, args=(que, ))
    p.start()
    beg = time.time()
    while time.time() - beg < total_timeout:
        if p.is_alive():
            time.sleep(.1)
        else:
            break
    else:
        p.terminate()
        p.join()
    mean_val = timeout * 1e3
    if not que.empty():
        mean_val = np.mean(np.array(que.get()))
    return mean_val


def batch_evaluate(s, bufs, target, dev_id, number=10, timeout=10.0):
    que = multi.Queue()
    total_timeout = timeout * number

    p = multi.Process(target=_evaluate, args=(s, bufs, target, dev_id, number, que))
    p.start()
    p.join(timeout=total_timeout)
    kill_child_processes(p.pid)
    p.terminate()
    p.join()
    mean_val = timeout * 1e3
    if not que.empty():
        mean_val = que.get(block=True)
    return mean_val


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


def __evaluate(s, bufs, target, dev_id, number=1, q=None):
    beg = time.time()
    for i in range(number):
        ctx = tvm.context(target, dev_id)
        tvm_arys = []
        for arg in bufs:
            shape = to_tuple(arg.shape)
            tmp = np.random.uniform(-10, 10, size=shape).astype(arg.dtype)
            tmp = tvm.nd.array(tmp, ctx)
            tvm_arys.append(tmp)
        try:
            func = tvm.build(s, bufs, target)
            func(*tvm_arys)
        except Exception as e:
            print("Oops")
            print(e)
    end = time.time()
    time_cost = (end - beg) * 1e3 / number
    if q:
        q.put(time_cost)
    return time_cost
