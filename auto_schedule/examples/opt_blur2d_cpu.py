import time
import os
import numpy as np
import argparse
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.test import test_graph_schedule_cpu_general_dx
from auto_schedule.train import Entity, train_op_schedule_cpu_general_dx


def run(M, N, k, model_path, epoch=5, sample_size=16, number=100, test=False):
    entities = []
    func = FUNC_TABLE["gaussian_blur2d"].func
    args = (M, N, k)
    entities.append(Entity("gaussian_blur2d", args))
    model_path = os.path.abspath(model_path)
    if not test:
        beg = time.time()
        train_op_schedule_cpu_general_dx(entities, epoch, sample_size, model_path)
        end = time.time()
        print("{}({}):".format("gaussian_blur2d", args))
        print("train done! use {}ms".format((end - beg) * 1e3))
    test_graph_schedule_cpu_general_dx(func, args, model_path, number=number)


def numpy_baseline(M, N, k, number=10):
    A = np.random.random([M, N])
    B = np.zeros([M, N])

    def blur(A, k, B):
        Apad = np.vstack([np.zeros([k//2, A.shape[1] + k // 2]), np.hstack([np.zeros([A.shape[0], k // 2]), A])])
        for i in range(k):
            for j in range(k):
                np.add(B, np.vstack([np.hstack([Apad[:A.shape[0], j:], np.zeros([A.shape[0], j])])[i:, :A.shape[1]], np.zeros([i, A.shape[1]])]), B)
        np.divide(B, k * k, B)

    beg = time.time()
    for i in range(number):
        blur(A, k, B)
    end = time.time()
    cost = (end - beg) * 1e3
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
        (1024, 1024, 7),
        (1024, 1024, 3),
        (2048, 2048, 3),
    ]

    names = [
        "1024_7",
        "1024_3",
        "2048_3",
    ]

    for i in range(len(arg_lst)):
        # model_path = "../models/opt_blur2d_" + names[i] + "_cpu_process.pkl"
        # entities = []
        # func = FUNC_TABLE["gaussian_blur2d"].func
        # args = arg_lst[i]
        # entities.append(Entity("gaussian_blur2d", args))
        # model_path = os.path.abspath(model_path)
        # train_op_schedule_cpu_general_dx(entities, 20, 25, model_path, logfile="process_blur_" + names[i] + "_cpu.txt", device="cuda:1")
        M, N, k = arg_lst[i]
        print(arg_lst[i], numpy_baseline(M, N, k))
