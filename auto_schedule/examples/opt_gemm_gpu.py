import time
import os
import torch
import argparse
from auto_schedule.test import test_graph_schedule_gpu_general_dx
from auto_schedule.train import Entity, train_op_schedule_gpu_general_dx


def run(batch_size, M, N, L, model_path, epoch=5, sample=16, number=100, test=False):
    entities = []
    args = (batch_size, M, N, L)
    entities.append(Entity("matmul_batch", args))
    model_path = os.path.abspath(model_path)
    if not test:
        beg = time.time()
        train_op_schedule_gpu_general_dx(entities, epoch, sample, model_path)
        end = time.time()
        print("{}({}):".format("matmul_batch", args))
        print("train done! use {}ms".format((end - beg) * 1e3))
    test_graph_schedule_gpu_general_dx(entities, model_path, sampling=True, number=number)


def pytorch_baseliine(batch_size, M, N, L, number=100):
    A = torch.rand((batch_size, M, N)).cuda()
    B = torch.rand((N, L)).cuda()
    beg = time.time()
    for i in range(number):
        C = A.matmul(B)
    end = time.time()
    cost = (end - beg) / number * 1e3
    return cost


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--train", help="train the model", action="store_true")
    # parser.add_argument("-p", "--pytorch", help="run pytorch baseline", action="store_true")
    # parser.add_argument("-a", "--auto_schedule", help="run auto-scheduler", action="store_true")
    # parser.add_argument("-n", "--number", help="number of tests", type=int, default=100)
    # parser.add_argument("-f", "--model_file_path", type=str, default="../logs/test_model.pkl")
    # parser.add_argument("--params", help="B,M,N,L", type=str, default="1,1024,1024,1024")
    # parser.add_argument("--epoch", type=int, default=5)
    # parser.add_argument("--sample", type=int, default=16)
    # args = parser.parse_args()
    # test = not args.train
    # use_torch = args.pytorch
    # use_auto = args.auto_schedule
    # try:
    #     params = [int(x) for x in args.params.split(",")]
    #     batch_size, M, N, L = params
    #     if use_torch:
    #         cost = pytorch_baseliine(batch_size, M, N, L, args.number)
    #         print("pytorch gemm use {}ms".format(cost))
    #     if use_auto:
    #         run(batch_size, M, N, L, args.model_file_path, args.epoch, args.sample, args.number, test)
    # except Exception as e:
    #     raise ValueError("Bad parameters, please refer to usage")

    # gemm
    arg_lst = [
        (1, 32, 32, 32),
        (1, 64, 64, 64),
        (1, 128, 128, 128),
        (1, 256, 256, 256),
        (1, 512, 512, 512),
        (1, 1024, 1024, 1024),
        (1, 2048, 2048, 2048),
        (1, 4096, 4096, 4096),
        (1, 1024, 32, 1024),
        (1, 32, 1024, 32),
    ]

    names = [
        "32_32_32_b1",
        "64_64_64_b1",
        "128_128_128_b1",
        "256_256_256_b1",
        "512_512_512_b1",
        "1024_1024_1024_b1",
        "2048_2048_2048_b1",
        "4096_4096_4096_b1",
        "1024_32_1024_b1",
        "32_1024_32_b1",
    ]
    for i in range(len(arg_lst)):
        model_path = "opt_gemm_" + names[i] + "_gpu.pkl"
        entities = []
        args = arg_lst[i]
        entities.append(Entity("matmul_batch", args))
        model_path = os.path.abspath(model_path)
        train_op_schedule_gpu_general_dx(entities, 20, 50, model_path, logfile="process_gemm_" + names[i] + "_gpu.txt", device="cuda:1")


