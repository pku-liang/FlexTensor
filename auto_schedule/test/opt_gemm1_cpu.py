import time
import os
import torch
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.test import test_graph_schedule_cpu_general_dx
from auto_schedule.train import Entity, train_op_schedule_cpu_general_dx


def run():
    entities = []
    func = FUNC_TABLE["matmul_batch"].func
    args = (1, 1024, 1024, 1024)
    entities.append(Entity(func, args))
    model_path = os.path.abspath("../models/opt_gemm1_cpu.pkl")
    beg = time.time()
    train_op_schedule_cpu_general_dx(entities, 5, 20, model_path)
    end = time.time()
    print("{}({}):".format("gemm", args))
    print("train done! use {}ms".format((end - beg) * 1e3))
    test_graph_schedule_cpu_general_dx(func, args, model_path)


def pytorch_baseliine():
    pass
    

if __name__ == "__main__":
    run()