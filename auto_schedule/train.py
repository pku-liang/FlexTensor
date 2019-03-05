import tvm
import torch
import numpy as np
import logging
from auto_schedule.training_examples import FUNC_TABLE
from auto_schedule.scheduler import graph_analysis, op_schedule_cpu_general_dx, evaluate
from auto_schedule.models import OpScheduleCPU


MAX_CPU = 8
C1 = 1
C2 = 1
LR = 0.01
MT = 0.7


class Entity(object):
    def __init__(self, func, args):
        self.func = func
        self.args = args


def train_op_schedule_cpu_general_dx(entities, epoch, batch_size):
    p = 0
    dim = 5
    num_sample = len(entities)
    model = OpScheduleCPU(dim, 3, 128)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=LR)
    model.train()
    for i in range(epoch):
        count = 0
        acc_loss = 0.0
        optimizer.zero_grad()
        model.train()
        while count < batch_size:
            func = entities[p].func
            args = entities[p].args
            ops, bufs = func(*args)
            s = tvm.create_schedule(ops)
            pre_cost = evaluate(s, bufs, "llvm", np.random.randint(0, MAX_CPU), 10, timeout=60.0)
            if not isinstance(ops, (list, tuple)):
                ops = [ops]
            bfs_order, down_graph = graph_analysis(ops)
            group_points = []
            for op in bfs_order:
                if not isinstance(op, tvm.tensor.ComputeOp):
                    continue
                # if able_inline(op, down_graph):
                # s[op].compute_inline()
                else:
                    group_points.append(op)
            device = torch.device("cpu:0")
            # improve = 1.0
            # for op in group_points:
            #     imp, _ = op_schedule_cpu_general_dx(dim, s, op, model, device, random=False)
            #     improve = improve * (1 + imp)
            improve, _ = op_schedule_cpu_general_dx(dim, s, group_points[0], model, device, random=False)
            post_cost = evaluate(s, bufs, "llvm", np.random.randint(0, MAX_CPU), 10, timeout=60.0)
            target = (pre_cost - post_cost) / pre_cost * 100
            loss_one = C1 * torch.pow(improve - target, 2)
            loss_two = C2 * torch.pow(100 - (improve + (target - improve).detach()), 2)
            acc_loss = acc_loss + loss_one + loss_two
            count += 1
            p = (p + 1) % num_sample
            print("    ({}, {}) part_one={}, part_two={}".format(float(improve.detach()), target, float(loss_one.detach()), float(loss_two.detach())))
        acc_loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), "models/test.pkl")
        print("epoch={}, loss={}\n".format(i + 1, float(acc_loss.detach())), flush=True)
        with open("train_test.log", "a") as f:
            f.write("epoch={}, loss={}\n".format(i + 1, float(acc_loss.detach())))
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
                # if able_inline(op, down_graph):
                # s[op].compute_inline()
                else:
                    group_points.append(op)
            device = torch.device("cpu:0")
            # improve = 1.0
            # for op in group_points:
            #     imp, _ = op_schedule_cpu_general_dx(dim, s, op, model, device, random=False)
            #     improve = improve * (1 + imp)
            op_schedule_cpu_general_dx(dim, s, group_points[0], model, device, random=False)
            time_cost = evaluate(s, bufs, "llvm", np.random.randint(0, MAX_CPU), 10, timeout=60.0)
            print("{}{} use {}ms\n".format(func.__name__, args, time_cost))


if __name__ == "__main__":
    entities = []
    # func = FUNC_TABLE["conv2d_channel_batch"].func
    # args = (1, 224, 224, 3, 3, 3, 4, 1, 1)
    # entities.append(Entity(func, args))
    func = FUNC_TABLE["matmul_batch"].func
    args = (1, 1024, 1024, 1024)
    entities.append(Entity(func, args))
    train_op_schedule_cpu_general_dx(entities, 20, 4)
