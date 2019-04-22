import time
import os
import numpy as np
import argparse
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.test import test_graph_schedule_cpu_general_dx
from auto_schedule.train import Entity, train_op_schedule_cpu_general_dx


def run(M, N, model_path, epoch=5, sample_size=16, number=100, test=False):
    entities = []
    func = FUNC_TABLE["outer_product"].func
    args = (M, N)
    entities.append(Entity("outer_product", args))
    model_path = os.path.abspath(model_path)
    if not test:
        beg = time.time()
        train_op_schedule_cpu_general_dx(entities, epoch, sample_size, model_path)
        end = time.time()
        print("{}({}):".format("outer_product", args))
        print("train done! use {}ms".format((end - beg) * 1e3))
    test_graph_schedule_cpu_general_dx(func, args, model_path, number=number)


def numpy_baseline(M, N, number=100):
    A = np.random.random(M)
    B = np.random.random(N)
    C = np.zeros([M, N])
    beg = time.time()
    for i in range(number):
        np.outer(A, B, C)
    end = time.time()
    cost = (end - beg) * 1e3 / number
    return cost


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--train", help="train the model", action="store_true")
    # parser.add_argument("-p", "--pytorch", help="run pytorch baseline", action="store_true")
    # parser.add_argument("-a", "--auto_schedule", help="run auto-scheduler", action="store_true")
    # parser.add_argument("-n", "--number", help="number of tests", type=int, default=100)
    # parser.add_argument("-f", "--model_file_path", type=str, default="../logs/test_model.pkl")
    # parser.add_argument("--params", help="N,H,W,L,C,k,K,stride,padding", type=str, default="1,14,14,14,512,3,512,1,1")
    # parser.add_argument("--epoch", type=int, default=5)
    # parser.add_argument("--sample", type=int, default=16)
    # args = parser.parse_args()
    # test = not args.train
    # use_torch = args.pytorch
    # use_auto = args.auto_schedule
    # try:
    #     params = [int(x) for x in args.params.split(",")]
    #     batch_size, height, width, depth, channel, kernel_size, output_channel, stride, padding = params
    #     if use_torch:
    #         pytorch_baseliine(batch_size, height, width, depth, channel, kernel_size, output_channel, stride, padding, args.number)
    #     if use_auto:
    #         run(batch_size, height, width, depth, channel, kernel_size, output_channel, stride, padding,
    #             args.model_file_path, args.epoch, args.sample, args.number, test)
    # except Exception as e:
    #     raise ValueError("Bad parameters, please refer to usage")
    arg_lst = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]

    names = [
        "512",
        "1024",
        "2048"
    ]

    for i in range(len(arg_lst)):
        model_path = "opt_outer_" + names[i] + "_cpu.pkl"
        entities = []
        func = FUNC_TABLE["outer_product"].func
        args = arg_lst[i]
        entities.append(Entity("outer_product", args))
        model_path = os.path.abspath(model_path)
        train_op_schedule_cpu_general_dx(entities, 10, 20, model_path, logfile="process_outer_" + names[i] + "_cpu.txt", device="cuda:1")
