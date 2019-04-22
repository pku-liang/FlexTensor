import tvm
import torch
import numpy as np
import os
import time
import torch.multiprocessing as _multi
multi = _multi.get_context("spawn")
from auto_schedule.models import OpScheduleCPUd5, OpScheduleGPUd5
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.scheduler import op_schedule_cpu_general_dx, graph_schedule_cpu_general_dx, \
    op_schedule_gpu_general_dx, graph_schedule_gpu_general_dx
from auto_schedule.measure import batch_evaluate, parallel_evaluate, _evaluate


CPU_NUM = os.cpu_count()
GPU_NUM = 4


def test_op_schedule_cpu():
    for name, compute in FUNC_TABLE.items():
        if name in ["conv3d", "conv3d_batch", "conv3d_channel", "conv3d_channel_batch"]:
            continue
        print("#################################")
        print(name)
        func = compute.func
        args = compute.args
        print(args)
        op, bufs = func(*args)
        dim = 5
        model = OpScheduleCPUd5(3, 128)
        s = tvm.create_schedule(op)
        # if torch.cuda.is_available():
        #     device= torch.device("cuda:0")
        #     model.cuda(device)
        # else:
        #     device = torch.device("cpu:0")

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(count_parameters(model))
        device = torch.device("cpu:0")
        lst, diary = op_schedule_cpu_general_dx(dim, s, op, model)
        print(tvm.lower(s, bufs, simple_mode=True))
        f = tvm.build(s, bufs, "llvm")
        cost = parallel_evaluate(s, bufs, "llvm", np.random.randint(0, CPU_NUM))
        msg = "{}({}) is {}ms, pass!\n".format(name, args, cost)
        print(msg)
        for ele in diary:
            print(ele)


def _test_op_schedule_gpu(timeout=10.0):
    proc = []
    q = multi.Queue()
    for i in range(10):
        p = multi.Process(target=test_op_schedule_gpu, args=(q,))
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
    cost_lst = []
    count = 0
    while not q.empty():
        cost_lst.append(q.get())
        count += 1
    while count < 10:
        cost_lst.append(timeout * 1e3)
        count += 1
    print(cost_lst)
    print(np.mean(np.array(cost_lst)))


def test_op_schedule_gpu(q=None):
    for name, compute in FUNC_TABLE.items():
        if name in ["conv3d", "conv3d_batch", "conv3d_channel", "conv3d_channel_batch", "gaussian_blur3x3"]:
            continue
        print("#################################")
        print(name)
        func = compute.func
        args = compute.args
        print(args)
        op, bufs = func(*args)
        dim = 5
        model = OpScheduleGPUd5(3, 128)
        s = tvm.create_schedule(op)
        if "conv" in name:
            print("check", name)
            A = op.input_tensors[0]
            s[A].compute_inline()
        # if torch.cuda.is_available():
        #     device= torch.device("cuda:0")
        #     model.cuda(device)
        # else:
        #     device = torch.device("cpu:0")
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(count_parameters(model))
        lst, diary = op_schedule_gpu_general_dx(dim, s, op, model, random=True)
        print(tvm.lower(s, bufs, simple_mode=True))
        cost = batch_evaluate(s, bufs, "cuda", np.random.randint(0, GPU_NUM), number=10)
        msg = "{}({}) is {}ms, pass!\n".format(name, args, cost)
        print(msg)
        # for ele in diary:
        #     print(ele)
        if q:
            q.put(cost)


def robust_test(round=100):
    for i in range(round):
        for name, compute in FUNC_TABLE.items():
            if name in ["conv3d", "conv3d_batch", "conv3d_channel", "conv3d_channel_batch"]:
                continue
            func = compute.func
            args = compute.args
            op, bufs = func(*args)
            dim = 5
            model = OpScheduleCPUd5(3, 32)
            s = tvm.create_schedule(op)
            lst, diary = op_schedule_cpu_general_dx(dim, s, op, model, random=True)
            try:
                stmt = tvm.lower(s, bufs, simple_mode=True)
                f = tvm.build(s, bufs, "llvm")
                cost = parallel_evaluate(s, bufs, "llvm", np.random.randint(0, CPU_NUM), timeout=0.003)
                msg = "{}({}) is {}ms, pass!\n".format(name, args, cost)
            except Exception as e:
                msg = "{}({}) error!\n{}\n{}\n".format(name, args, str(e), diary)
            with open("test_log.log", "a") as f:
                f.write(msg)


def test_graph_schedule_cpu_general_dx(func, args, model_path, random=False, sampling=False, number=10, timeout=10.0):
    print("#################################")
    ops, bufs = func(*args)
    dim = 5
    s = tvm.create_schedule(ops)
    graph_schedule_cpu_general_dx(dim, s, ops, model_path, random=random, sampling=sampling)
    cost = batch_evaluate(s, bufs, "llvm", np.random.randint(0, CPU_NUM), number=number, timeout=timeout)
    msg = "Optimial time cost is {}ms, pass!\n".format(cost)
    print(msg)
    return cost


def test_graph_schedule_gpu_general_dx(entities, model_path, random=False, sampling=False, timeout=10.0, trial=30, number=10):

    def _inner(sampling=False):
        queue = multi.Queue()
        proc = multi.Process(target=_eval_gpu, args=(dim, entity, model_path, queue, random, sampling, 1, number))
        proc.start()
        proc.join(timeout=timeout * number)

        if not queue.empty():
            cost = queue.get(block=True)
        else:
            cost = timeout * 1e3

        proc.terminate()
        try:
            proc.join()
        except Exception as e:
            pass
        return cost
    print("#################################")
    dim = 5
    cost_lst = []
    for entity in entities:
        best_cost = _inner(False)   # non-sampling case
        if sampling:
            for i in range(trial):
                cost = _inner(True)
                if cost < best_cost:
                    best_cost = cost
        cost_lst.append(best_cost)
        msg = "Optimial time cost for {} is {}ms, pass!\n".format(entity.func_name, best_cost)
        print(msg)
    return cost_lst


def _eval_gpu(dim, entity, model_path, queue, random, sampling, trial=10, number=10):
    func_name = entity.func_name
    func = FUNC_TABLE[func_name].func
    args = entity.args
    best_time = float("+inf")
    model = OpScheduleGPUd5(3, 128)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    model.eval()
    for i in range(trial):
        ops, bufs = func(*args)
        s = tvm.create_schedule(ops)
        graph_schedule_gpu_general_dx(dim, s, ops, model_path, random=random, sampling=sampling)
        # print(tvm.lower(s, bufs, simple_mode=True))
        try:
            time_cost = _evaluate(s, bufs, "cuda", dev_id=np.random.randint(0, GPU_NUM), number=number)
            if time_cost < best_time:
                best_time = time_cost
        except Exception as e:
            pass
    if best_time != float("+inf"):
        queue.put(best_time)
    print(best_time)


if __name__ == "__main__":
    for i in range(100):
        test_op_schedule_gpu()



