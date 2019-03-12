import tvm
import torch
import numpy as np
import logging
import heapq
import time
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.scheduler import graph_analysis, op_schedule_cpu_general_dx, evaluate, able_inline
from auto_schedule.models import OpScheduleCPUd5
from auto_schedule.test import test_graph_schedule_cpu_general_dx


MAX_CPU = 8
C1 = 1
C2 = 1
LR = 0.002
MT = 0.7


class Entity(object):
    def __init__(self, func, args):
        self.func = func
        self.args = args


def train_op_schedule_cpu_general_dx(entities, epoch, batch_size, path, loop_num=100, loop_size=32):
    p = 0
    dim = 5
    num_sample = len(entities)
    model = OpScheduleCPUd5(3, 128)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=LR)
    model.train()
    dataset = []
    for i in range(epoch):
        count = 0
        optimizer.zero_grad()
        model.train()
        while count < batch_size:
            func = entities[p].func
            args = entities[p].args
            ops, bufs = func(*args)
            s = tvm.create_schedule(ops)
            pre_cost = evaluate(s, bufs, "llvm", np.random.randint(0, MAX_CPU), 1, timeout=10.0)
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
            y_dict, _ = op_schedule_cpu_general_dx(dim, s, group_points[0], model, random=False, sampling=True)
            post_cost = evaluate(s, bufs, "llvm", np.random.randint(0, MAX_CPU), 10, timeout=10.0)
            print("pre=", pre_cost, "ms post=", post_cost, "ms")
            target = (pre_cost - post_cost) / pre_cost * 10
            data = dict()
            for name, value in y_dict.items():
                if isinstance(value, list):
                    tmp = []
                    for v in value:
                        tmp.append(v.detach())
                    data[name] = tmp
                else:
                    data[name] = value.detach()
            if target > 0:
                dataset.append((p, data, post_cost))
            else:
                dataset.append((p, data, post_cost))
            count += 1
            p = (p + 1) % num_sample
        # test if can approximate the targets
        for r in range(loop_num):
            acc_loss = 0.0
            for inner in range(loop_size):
                for (q, data, time_cost) in heapq.nsmallest(1, dataset, key=lambda x: x[-1]):
                    func = entities[q].func
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
                    y_dict, _ = op_schedule_cpu_general_dx(dim, s, group_points[0], model, random=False, sampling=False)
                    # spatial loss
                    spatial_loss = 0.0
                    for i in range(dim):
                        spatial_loss = spatial_loss + torch.nn.functional.binary_cross_entropy(y_dict["spatial"][i], data["spatial"][i])
                        if r % 10 == 0 and inner == 9:
                            print("    spatial:", float(spatial_loss.detach()))
                    # reduce_loss
                    reduce_loss = 0.0
                    for i in range(dim):
                        reduce_loss = reduce_loss + torch.nn.functional.binary_cross_entropy(y_dict["reduce"][i], data["reduce"][i])
                        if r % 10 == 0 and inner == 9:
                            print("    reduce:", float(reduce_loss.detach()))
                    # parallel_loss
                    parallel_loss = torch.nn.functional.binary_cross_entropy(y_dict["parallel"], data["parallel"])
                    if r % 10 == 0 and inner == 9:
                        print("    parallel:", float(parallel_loss.detach()))
                    # reorder_one loss
                    reorder_one_loss = torch.nn.functional.binary_cross_entropy(y_dict["reorder_one"], data["reorder_one"])
                    if r % 10 == 0 and inner == 9:
                        print("    reorder_one:", float(reorder_one_loss.detach()))
                    # reorder_two loss
                    reorder_two_loss = torch.nn.functional.binary_cross_entropy(y_dict["reorder_two"], data["reorder_two"])
                    if r % 10 == 0 and inner == 9:
                        print("    reorder_two:", float(reorder_two_loss.detach()))
                    # reorder_three loss
                    reorder_three_loss = torch.nn.functional.binary_cross_entropy(y_dict["reorder_three"], data["reorder_three"])
                    if r % 10 == 0 and inner == 9:
                        print("    reorder_three:", float(reorder_three_loss.detach()))
                    # accumulate loss
                    acc_loss = acc_loss + spatial_loss + reduce_loss + parallel_loss + reorder_one_loss + reorder_two_loss + reorder_three_loss
            acc_loss.backward()
            if r % 10 == 0:
                # for param in model.sard1.parameters():
                #     print(param.grad)
                torch.save(model.state_dict(), path)
                print("epoch={}, r={}, loss={}\n".format(i + 1, r, float(acc_loss.detach())), flush=True)
                # with open("train_test.log", "a") as f:
                #     f.write("epoch={}, loss={}\n".format(i + 1, float(acc_loss.detach())))
            optimizer.step()

        model.eval()
        for entity in entities:
            func = entity.func
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
            op_schedule_cpu_general_dx(dim, s, group_points[0], model, random=False, sampling=False)
            print(tvm.lower(s, bufs, simple_mode=True))
            time_cost = evaluate(s, bufs, "llvm", np.random.randint(0, MAX_CPU), 5, timeout=10.0)
            print("{}{} use {}ms\n".format(func.__name__, args, time_cost))


if __name__ == "__main__":
    entities = []
    func = FUNC_TABLE["conv2d_channel_batch"].func
    args = (1, 14, 14, 256, 3, 3, 512, 1, 1)
    entities.append(Entity(func, args))
    # func = FUNC_TABLE["matmul_batch"].func
    # args = (1, 1024, 1024, 1024)
    # entities.append(Entity(func, args))
    beg = time.time()
    train_op_schedule_cpu_general_dx(entities, 5, 8, "models/test.pkl")
    end = time.time()
    print("train done! use {}ms".format((end - beg) * 1e3))
    test_graph_schedule_cpu_general_dx(func, args, "./models/test.pkl")
