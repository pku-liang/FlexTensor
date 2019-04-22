import os
import pickle
import tvm
import torch
import torch.multiprocessing as _multi
multi = _multi.get_context("spawn")
from functools import partial
import numpy as np
import logging
import heapq
import time
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.scheduler import graph_analysis, able_inline, op_schedule_cpu_general_dx
from auto_schedule.testing import op_schedule_gpu_general_dx
from auto_schedule.measure import serial_evaluate, batch_evaluate, _evaluate
from auto_schedule.models import OpScheduleCPUd5, OpScheduleGPUd5
from auto_schedule.test import test_graph_schedule_gpu_general_dx
from auto_schedule.utils import to_tuple, free_cuda


MAX_CPU = os.cpu_count()
MAX_GPU = 3
C1 = 1
C2 = 1
LR = 0.002
MT = 0.7


class Entity(object):
    def __init__(self, func_name, args):
        self.func_name = func_name
        self.args = args


def train_op_schedule_cpu_general_dx(entities, epoch, batch_size, path, loop_num=100, loop_size=16,
                                     stack_size=20, logfile="temp.log", device="cuda:0"):
    dim = 5
    timeout = 15.0
    num_sample = len(entities)
    device = torch.device(device)
    model = OpScheduleCPUd5(3, 128, device)
    # load or initialize parameter file
    if os.path.exists(path) and os.path.isfile(path):
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    else:
        torch.save(model.state_dict(), path)
    model.to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=LR)
    model.train()
    # maintain a dataset for each function
    datasets = [[] for i in range(num_sample)]

    train_beg_time = time.time()
    with open(logfile, "a") as f:
        f.write("New log\ntime: {}".format(train_beg_time))
    perf_before = dict()
    perf_before_dump = False
    model.train()
    print("Scheduling begins...parameters in path {}\n    logs to{}".format(path, logfile))
    for i in range(epoch):
        optimizer.zero_grad()
        for batch in range(batch_size):
            for p in range(num_sample):
                func_name = entities[p].func_name
                func = FUNC_TABLE[func_name].func
                args = entities[p].args
                ops, bufs = func(*args)
                s = tvm.create_schedule(ops)
                # get the performance before scheduling
                # only run one time
                entity_key = "{}:{}".format(func_name, args)
                if entity_key not in perf_before:
                    pre_cost = serial_evaluate(s, bufs, "llvm", np.random.randint(0, MAX_CPU), 10, timeout=timeout)
                    perf_before[entity_key] = pre_cost
                if not isinstance(ops, (list, tuple)):
                    ops = [ops]
                bfs_order, down_graph = graph_analysis(ops)
                group_points = []
                for op in bfs_order:
                    if not isinstance(op, tvm.tensor.ComputeOp):
                        continue
                    if able_inline(op, down_graph):
                        s[op].compute_inline()
                    else:
                        group_points.append(op)
                if len(group_points) > 1:
                    raise RuntimeError("Not support more than one compute")
                for j, point in enumerate(group_points):
                    y_dict, y_diary = op_schedule_cpu_general_dx(dim, s, point, model, random=np.random.random() < 0.2, sampling=True)
                    post_cost = serial_evaluate(s, bufs, "llvm", np.random.randint(0, MAX_CPU), 10, timeout=timeout)
                    data = dict()
                    for name, value in y_dict.items():
                        if isinstance(value, list):
                            tmp = []
                            for v in value:
                                tmp.append(v.detach())
                            data[name] = (tmp, y_diary[name])   # the data record schedule decisions
                        else:
                            data[name] = (value.detach(), y_diary[name])
                        # record  (point No. , sch data, time cost)
                        datasets[p].append((j, data, post_cost))
        # record performance before scheduling
        # only run one time
        if not perf_before_dump:
            with open(logfile, "a") as f:
                logs = "performance before scheduling:\n"
                f.write(logs)
                for key, perf in perf_before.items():
                    logs = "{}: {}\n".format(key, perf)
                    f.write(logs)
                f.write("\n")
            perf_before_dump = True
        # control the size of dataset and record best cases
        cur_time = time.time()
        with open(logfile, "a") as f:
            for j in range(num_sample):
                datasets[j] = heapq.nsmallest(stack_size, datasets[j], key=lambda x: x[-1])
                entity_key = "{}:{}".format(entities[j].func_name, entities[j].args)
                duration = cur_time - train_beg_time
                logs = "epoch {}/{}| {} best perf {}| [{}s]\n".format(i+1, epoch, entity_key, datasets[j][0][-1], duration)
                f.write(logs)
                logs = "schedule {}\n".format(entity_key)
                for name, val in datasets[j][0][1].items():    # find the diary, this is ugly now, change later
                    logs = logs + "{}: {}\n".format(name, val[1])
                logs = logs + "\n"
                f.write(logs)
        # train the parameters
        for r in range(loop_num):
            acc_loss = 0.0
            for inner in range(loop_size):
                for q in range(num_sample):
                    func_name = entities[q].func_name
                    func = FUNC_TABLE[func_name].func
                    args = entities[q].args
                    for (point_num, data, time_cost) in datasets[q][:1]:
                        ops, bufs = func(*args)
                        s = tvm.create_schedule(ops)
                        if not isinstance(ops, (list, tuple)):
                            ops = [ops]
                        bfs_order, down_graph = graph_analysis(ops)
                        group_points = []
                        for op in bfs_order:
                            if not isinstance(op, tvm.tensor.ComputeOp):
                                continue
                            if able_inline(op, down_graph):
                                s[op].compute_inline()
                            else:
                                group_points.append(op)
                        y_dict, _ = op_schedule_cpu_general_dx(dim, s, group_points[point_num], model, random=False, sampling=False)
                        # spatial loss
                        spatial_loss = 0.0
                        for j in range(dim):
                            spatial_loss = spatial_loss + torch.nn.functional\
                                .binary_cross_entropy(y_dict["spatial"][j], data["spatial"][0][j])
                        # reduce_loss
                        reduce_loss = 0.0
                        for j in range(dim):
                            reduce_loss = reduce_loss + torch.nn.functional\
                                .binary_cross_entropy(y_dict["reduce"][j], data["reduce"][0][j])
                        # parallel_loss
                        parallel_loss = torch.nn.functional\
                            .binary_cross_entropy(y_dict["parallel"], data["parallel"][0])
                        # reorder_one loss
                        reorder_one_loss = torch.nn.functional\
                            .binary_cross_entropy(y_dict["reorder_one"], data["reorder_one"][0])
                        # reorder_two loss
                        reorder_two_loss = torch.nn.functional\
                            .binary_cross_entropy(y_dict["reorder_two"], data["reorder_two"][0])
                        # reorder_three loss
                        reorder_three_loss = torch.nn.functional\
                            .binary_cross_entropy(y_dict["reorder_three"], data["reorder_three"][0])
                        # accumulate loss
                        acc_loss = acc_loss + spatial_loss + reduce_loss + parallel_loss + reorder_one_loss \
                                   + reorder_two_loss + reorder_three_loss
            acc_loss.backward()
            if r % 10 == 0:
                torch.save(model.state_dict(), path)
                logs = "epoch={}, r={}, loss={}\n".format(i + 1, r, float(acc_loss.detach()))
                with open(logfile, "a") as f:
                    f.write(logs)
            optimizer.step()
        with open(logfile, "a") as f:
            f.write("\n")
    print("All done.")


def detach_data(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    elif isinstance(data, list):
        ret = []
        for item in data:
            ret.append(detach_data(item))
        return ret
    elif isinstance(data, dict):
        ret = dict()
        for key, val in data.items():
            ret[key] = detach_data(val)
        return ret
    else:
        raise ValueError("Detach data only support [torch.Tensor, list, dict], but get {}".format(type(data)))


def train_op_schedule_gpu_general_dx(entities, epoch, batch_size, model_path, loop_num=50, loop_size=8,
                                     stack_size=20, logfile="temp.log"):
    dim = 5
    timeout = 10.0
    fail_data_num = 0
    num_entity = len(entities)
    datasets = [[] for i in range(num_entity)]
    train_beg_time = time.time()
    print("Schedule begins...parameters in {}\n    logs to {}".format(model_path, logfile), flush=True)
    with open(logfile, "a") as f:
        f.write("New log [{}]\n".format(train_beg_time))
    free_device = free_cuda()
    if free_device:
        use_device = "cuda:{}".format(free_device[0])
    else:
        print("Warning: no available GPU for training, using GPU...")
        use_device = "cpu:0"
    print("train_device", use_device)
    device = torch.device(use_device)
    model = OpScheduleGPUd5(3, 128, device)
    if os.path.exists(model_path):
        print("INFO: using model file from {}".format(model_path))
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    else:
        torch.save(model.state_dict(), model_path)
    model.to(device)
    model.train()
    for ep_num in range(epoch):
        proc_pool = []
        que_pool = []
        y_pool = []
        diary_pool = []
        arg_lst = []
        time_cost_pool = []
        for batch in range(batch_size):
            for p_entity in range(num_entity):
                entity = entities[p_entity]
                func = FUNC_TABLE[entity.func_name].func
                args = entity.args
                ops, bufs = func(*args)
                s = tvm.create_schedule(ops)
                if not isinstance(ops, (list, tuple)):
                    ops = [ops]
                bfs_order, down_graph = graph_analysis(ops)
                group_points = []
                for op in bfs_order:
                    if not isinstance(op, tvm.tensor.ComputeOp):
                        continue
                    if able_inline(op, down_graph):
                        s[op].compute_inline()
                    else:
                        group_points.append(op)
                y, diary = op_schedule_gpu_general_dx(dim, s, group_points[0], model, random=np.random.random()<0.5, sampling=True)
                y_pool.append(detach_data(y))
                diary_pool.append(diary)
                arg_lst.append((dim, entity, diary))
                time_cost_pool.append([p_entity, float("inf")])
        target_func = partial(call_with_timeout, _get_data_gpu, timeout)
        for arg in arg_lst:
            q = multi.Queue(2)
            p = multi.Process(target=target_func, args=(q, *arg))
            proc_pool.append(p)
            que_pool.append(q)
            p.start()
        wait_start = time.time()
        while time.time() - wait_start < timeout:
            if any([p.is_alive() for p in proc_pool]):
                time.sleep(.1)
            else:
                break
        for count_proc, q in enumerate(que_pool):
            if not q.empty():
                res = q.get(block=True)
                if not isinstance(res, multi.TimeoutError):
                    time_cost_pool[count_proc][1] = res
                else:
                    fail_data_num += 1
            p = proc_pool[count_proc]
            if p.is_alive():
                p.terminate()
            p.join()
            q.close()
            q.join_thread()
            del q
            del p
        for count_y, (id, time_cost) in enumerate(time_cost_pool):
            print("time_cost", time_cost)
            if time_cost < float("inf"):
                datasets[id].append((y_pool[count_y], diary_pool[count_y], time_cost))
        # control dataset
        cur_time = time.time()
        with open(logfile, "a") as f:
            for j in range(num_entity):
                datasets[j] = heapq.nsmallest(stack_size, datasets[j], key=lambda x: x[-1])
                entity_key = "{}:{}".format(entities[j].func_name, entities[j].args)
                duration = cur_time - train_beg_time
                if datasets[j]:
                    logs = "epoch {}/{}| {} best perf {}| [{}s]\n".format(ep_num + 1, epoch, entity_key, datasets[j][0][-1],
                                                                          duration)
                    f.write(logs)
                    logs = "schedule {}\n".format(entity_key)
                    for name, val in datasets[j][0][1].items():  # find the diary, this is ugly now, change later
                        logs = logs + "{}: {}\n".format(name, val)
                    logs = logs + "\n"
                    f.write(logs)
                else:
                    logs = "epoch {}/{}| {} best perf not yet get| [{}s]\n".format(ep_num + 1, epoch, entity_key, duration)
                    f.write(logs)
        # train on dataset
        model.train()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=LR)
        _train_gpu(dim, entities, model, optimizer, datasets, loop_num, loop_size, logfile)
        torch.save(model.state_dict(), model_path)
    print("All done.")


def call_with_timeout(func, timeout, resq, *args):
    proc = multi.Process(target=func, args=(*args, resq))
    proc.start()
    proc.join(timeout=timeout)
    proc.terminate()
    proc.join()
    resq.put(multi.TimeoutError())


def _get_data_gpu(dim, entity, outer_diary, q):
    func = FUNC_TABLE[entity.func_name].func
    args = entity.args
    ops, bufs = func(*args)
    s = tvm.create_schedule(ops)
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    bfs_order, down_graph = graph_analysis(ops)
    group_points = []
    for op in bfs_order:
        if not isinstance(op, tvm.tensor.ComputeOp):
            continue
        if able_inline(op, down_graph):
            s[op].compute_inline()
        else:
            group_points.append(op)
    op_schedule_gpu_general_dx(dim, s, group_points[0], outer_diary=outer_diary)
    free_device = free_cuda()
    if free_device:
        dev_id = free_device[0]
    else:
        print("Warning: no available GPU for test, trying to use cuda:0")
        dev_id = 0
    try:
        time_cost = _evaluate(s, bufs, "cuda", dev_id=dev_id, number=10)
    except Exception as e:
        time_cost = float("inf")
    q.put(time_cost)


def _train_gpu(dim, entities, model, optimizer, datasets, loop_num, loop_size, logfile):
    gamma = 0.9
    for r in range(loop_num):
        acc_loss = 0.0
        has_data = False
        for inner in range(loop_size):
            for q in range(len(entities)):
                factor = 1.0
                for (target_y, target_d, time_cost) in datasets[q]:
                    has_data = True
                    func_name = entities[q].func_name
                    func = FUNC_TABLE[func_name].func
                    args = entities[q].args
                    ops, bufs = func(*args)
                    s = tvm.create_schedule(ops)
                    if not isinstance(ops, (list, tuple)):
                        ops = [ops]
                    bfs_order, down_graph = graph_analysis(ops)
                    group_points = []
                    for op in bfs_order:
                        if not isinstance(op, tvm.tensor.ComputeOp):
                            continue
                        if able_inline(op, down_graph):
                            s[op].compute_inline()
                        else:
                            group_points.append(op)
                    point = group_points[0]
                    y_dict, diary = op_schedule_gpu_general_dx(dim, s, point, model, random=False, sampling=False)
                    # spatial loss one
                    spatial_loss = 0.0
                    # spatial part one
                    for i in range(dim):
                        spatial_loss = spatial_loss + torch.nn.functional\
                            .binary_cross_entropy(y_dict["spatial_one"][i], target_y["spatial_one"][i])
                    # reduce_loss
                    reduce_loss = 0.0
                    for i in range(dim):
                        reduce_loss = reduce_loss + torch.nn.functional\
                            .binary_cross_entropy(y_dict["reduce"][i], target_y["reduce"][i])
                    # reorder_one loss
                    reorder_one_loss = torch.nn.functional\
                        .binary_cross_entropy(y_dict["reorder_one"], target_y["reorder_one"])
                    # reorder_two loss
                    reorder_two_loss = torch.nn.functional\
                        .binary_cross_entropy(y_dict["reorder_two"], target_y["reorder_two"])
                    # reorder_three loss
                    reorder_three_loss = torch.nn.functional\
                        .binary_cross_entropy(y_dict["reorder_three"], target_y["reorder_three"])
                    # accumulate loss
                    acc_loss = acc_loss + factor * (spatial_loss + reduce_loss + reorder_one_loss
                                                    + reorder_two_loss + reorder_three_loss)
                    factor *= gamma
        if has_data:
            acc_loss.backward()
            if r % 10 == 0:
                logs = "loss={}\n".format(float(acc_loss.detach()))
                with open(logfile, "a") as f:
                    f.write(logs)
            optimizer.step()
        else:
            with open(logfile, "a") as f:
                f.write("no data\n")
    with open(logfile, "a") as f:
        f.write("\n")


def _eval_gpu(dim, entity, model_path, queue, trial=10, number=10):
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
        if not isinstance(ops, (list, tuple)):
            ops = [ops]
        bfs_order, down_graph = graph_analysis(ops)
        group_points = []
        for op in bfs_order:
            if not isinstance(op, tvm.tensor.ComputeOp):
                continue
            if able_inline(op, down_graph):
                s[op].compute_inline()
            else:
                group_points.append(op)
        op_schedule_gpu_general_dx(dim, s, group_points[0], model, random=False, sampling=True)
        try:
            time_cost = _evaluate(s, bufs, "cuda", dev_id=np.random.randint(0, MAX_GPU), number=number)
            if time_cost < best_time:
                best_time = time_cost
        except Exception as e:
            pass
    queue.put(best_time)


if __name__ == "__main__":
    entities = []
    # func = FUNC_TABLE["conv2d_channel_batch"].func
    # args = (1, 14, 14, 256, 3, 3, 512, 1, 1)
    # entities.append(Entity("conv2d_channel_batch", args))
    func = FUNC_TABLE["matmul_batch"].func
    args = (1, 1024, 1024, 1024)
    entities.append(Entity("matmul_batch", args))
    beg = time.time()
    train_op_schedule_gpu_general_dx(entities, 5, 16, "models/test_gemm_gpu.pkl")
    end = time.time()
    print("train done! use {}ms".format((end - beg) * 1e3))
    test_graph_schedule_gpu_general_dx(entities, "./models/test_gemm_gpu.pkl", sampling=True)
