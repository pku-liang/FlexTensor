import time
import os
import torch
import argparse
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.test import test_graph_schedule_cpu_general_dx
from auto_schedule.train import Entity, train_op_schedule_cpu_general_dx


def run(test=False):
    entities = []
    func = FUNC_TABLE["matmul_batch"].func
    args = (1, 1024, 1024, 1024)
    entities.append(Entity(func, args))
    model_path = os.path.abspath("../models/opt_gemm1_cpu.pkl")
    if not test:
        beg = time.time()
        train_op_schedule_cpu_general_dx(entities, 5, 16, model_path)
        end = time.time()
        print("{}({}):".format("matmul_batch", args))
        print("train done! use {}ms".format((end - beg) * 1e3))
    test_graph_schedule_cpu_general_dx(func, args, model_path)


def pytorch_baseliine():
    A = torch.rand((1, 1024, 1024))
    B = torch.rand((1024, 1024))
    beg = time.time()
    for i in range(100):
        C = A.matmul(B)
    end = time.time()
    print("pytorch gemm use {}ms".format((end - beg) / 100 * 1e3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="train the model", action="store_true")
    parser.add_argument("-p", "--pytorch", help="run pytorch baseline", action="store_true")
    parser.add_argument("-a", "--auto_schedule", help="run auto-scheduler", action="store_true")
    args = parser.parse_args()
    test = not args.train
    use_torch = args.pytorch
    use_auto = args.auto_schedule
    if use_torch:
        pytorch_baseliine()
    if use_auto:
        run(test)