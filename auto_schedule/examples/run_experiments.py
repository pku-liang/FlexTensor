import os
import time
from auto_schedule.train import train_op_schedule_cpu_general_dx, train_op_schedule_gpu_general_dx, Entity
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.test import test_graph_schedule_cpu_general_dx, test_graph_schedule_gpu_general_dx


if __name__ == "__main__":
    # convolutions
    arg_lst = [
        (1, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (8, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (64, 7, 7, 1024, 3, 3, 1024, 1, 1),
        # (256, 7, 7, 1024, 3, 3, 1024, 1, 1),
        (1, 14, 14, 1024, 1, 1, 512, 1, 0),
        (1, 28, 28, 256, 3, 3, 512, 1, 1),
        (1, 28, 28, 512, 1, 1, 256, 1, 0),
        (1, 56, 56, 128, 3, 3, 256, 1, 1),
        (1, 56, 56, 192, 1, 1, 128, 1, 0),
        (1, 112, 112, 64, 3, 3, 192, 1, 1),
        (1, 448, 448, 3, 7, 7, 64, 2, 3)
    ]
    names = [
        "yolo24_b1",
        # "yolo24_b8",
        # "yolo24_b64",
        # "yolo24_b256",
        "yolo19_b1",
        "yolo10_b1",
        "yolo7_b1",
        "yolo4_b1",
        "yolo3_b1",
        "yolo2_b1",
        "yolo1_b1"
    ]

    # pytorch baseline
    # for i in range(len(arg_lst)):
    #     print(arg_lst[i])
    #     N, H, W, C, kernel_size, _, K, stride, padding = arg_lst[i]
    #     cost10 = pytorch_baseliine(N, H, W, C, kernel_size, K, stride, padding, 10)
    #     cost100 = pytorch_baseliine(N, H, W, C, kernel_size, K, stride, padding, 100)
    #     cost1000 = pytorch_baseliine(N, H, W, C, kernel_size, K, stride, padding, 1000)
    #     logs = "[10:{},100:{},1000:{}]\n".format(cost10, cost100, cost1000)
    #     with open("logs_pytorch_conv_gpu.txt", "a") as f:
    #         f.write(logs)

    for i in range(len(arg_lst)):
        model_path = "../models/opt_conv_" + names[i] + "_gpu_new.pkl"
        entities = []
        func = FUNC_TABLE["conv2d_channel_batch"].func
        args = arg_lst[i]
        entities.append(Entity("conv2d_channel_batch", args))
        model_path = os.path.abspath(model_path)
        beg = time.time()
        train_op_schedule_gpu_general_dx(entities, 10, 32, model_path)
        end = time.time()
        logs = "{}({}):\n".format("conv2d_channel_batch", args) + "train done! use {}ms\n".format((end - beg) * 1e3)
        cost10 = test_graph_schedule_gpu_general_dx(entities, model_path, number=10)[0]
        cost100 = test_graph_schedule_gpu_general_dx(entities, model_path, number=100)[0]
        cost1000 = test_graph_schedule_gpu_general_dx(entities, model_path, number=1000)[0]
        logs = logs + "[10:{},100:{},1000:{}]\n".format(cost10, cost100, cost1000)
        with open("logs_conv_gpu_new.txt", "a") as f:
            f.write(logs)

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
        (1, 32, 256, 1024),
        (32, 32, 32, 32),
        (128, 32, 32, 32),
        (256, 32, 32, 32),
        (1024, 16, 16, 16)
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
            "32_256_1024_b1",
            "32_32_32_b32",
            "32_32_32_b128",
            "32_32_32_b256",
            "16_16_16_b1024"
        ]

    # pytorch baseline
    # for i in range(len(arg_lst)):
    #     batch_size, M, N, L = arg_lst[i]
    #     cost10 = pytorch_baseliine(batch_size, M, N, L, 10)
    #     cost100 = pytorch_baseliine(batch_size, M, N, L, 100)
    #     cost1000 = pytorch_baseliine(batch_size, M, N, L, 1000)
    #     logs = "{} [10:{},100:{},1000:{}]\n".format(names[i], cost10, cost100, cost1000)
    #     with open("logs_pytorch_gemm_cpu.txt", "a") as f:
    #         f.write(logs)

    # numpy baseline
    # import numpy as np
    #     for i in range(len(arg_lst)):
    #         batch_size, M, N, L = arg_lst[i]
    #         A = np.random.random([batch_size, M, N])
    #         B = np.random.random([N, L])
    #         beg = time.time()
    #         for j in range(10):
    #             C = np.matmul(A, B)
    #         end = time.time()
    #         cost10 = (end - beg) * 1e3 / 10
    #         for j in range(100):
    #             C = np.matmul(A, B)
    #         end = time.time()
    #         cost100 = (end - beg) * 1e3 / 100
    #         for j in range(1000):
    #             C = np.matmul(A, B)
    #         end = time.time()
    #         cost1000 = (end - beg) * 1e3 / 1000
    #         logs = "{} [10:{},100:{},1000:{}]\n".format(names[i], cost10, cost100, cost1000)
    #         with open("logs_numpy_gemm.txt", "a") as f:
    #             f.write(logs)

    for i in range(len(arg_lst)):
        model_path = "../models/opt_gemm_" + names[i] + "_gpu_new.pkl"
        entities = []
        func = FUNC_TABLE["matmul_batch"].func
        args = arg_lst[i]
        entities.append(Entity("matmul_batch", args))
        model_path = os.path.abspath(model_path)
        beg = time.time()
        train_op_schedule_gpu_general_dx(entities, 10, 32, model_path)
        end = time.time()
        logs = "{}({}):\n".format("matmul_batch", args) + "train done! use {}ms\n".format((end - beg) * 1e3)
        cost10 = test_graph_schedule_gpu_general_dx(entities, model_path, number=10)[0]
        cost100 = test_graph_schedule_gpu_general_dx(entities, model_path, number=100)[0]
        cost1000 = test_graph_schedule_gpu_general_dx(entities, model_path, number=1000)[0]
        logs = logs + "[10:{},100:{},1000:{}]\n".format(cost10, cost100, cost1000)
        with open("logs_gemm_gpu_new.txt", "a") as f:
            f.write(logs)

    # for i in range(len(arg_lst)):
    #     batch_size, M, N, L = arg_lst[i]
    #     cost10 = pytorch_baseliine(batch_size, M, N, L, 10)
    #     cost100 = pytorch_baseliine(batch_size, M, N, L, 100)
    #     cost1000 = pytorch_baseliine(batch_size, M, N, L, 1000)
    #     logs = "{} [10:{},100:{},1000:{}]\n".format(names[i], cost10, cost100, cost1000)
    #     with open("logs_pytorch_gemm_gpu.txt", "a") as f:
    #         f.write(logs)

    # conv3d
    # arg_lst = [
    #         (1, 7, 7, 7, 1024, 3, 3, 3, 1024, 1, 1),
    #         (1, 14, 14, 14, 1024, 1, 1, 1, 512, 1, 0),
    #         (1, 28, 28, 28, 256, 3, 3, 3, 512, 1, 1),
    #         (1, 28, 28, 28, 512, 1, 1, 1, 256, 1, 0),
    #         (1, 56, 56, 56, 192, 1, 1, 1, 128, 1, 0),
    #         (1, 112, 112, 10, 64, 3, 3, 3, 192, 1, 1),
    #         (1, 448, 448, 10, 3, 7, 7, 7, 64, 2, 3)
    #     ]
    #
    # names = [
    #         "yolo24_b1",
    #         "yolo19_b1",
    #         "yolo10_b1",
    #         "yolo7_b1",
    #         "yolo3_b1",
    #         "yolo2_b1",
    #         "yolo1_b1"
    #     ]
    #
    # for i in range(len(arg_lst)):
    #     batch_size, height, width, depth, channel, kernel_size, _, _, output_channel, stride, padding = arg_lst[i]
    #     model_path = "../models/opt_conv3d_" + names[i] + "_cpu.pkl"
    #     entities = []
    #     func = FUNC_TABLE["conv3d_channel_batch"].func
    #     args = arg_lst[i]
    #     entities.append(Entity("conv3d_channel_batch", args))
    #     model_path = os.path.abspath(model_path)
    #     beg = time.time()
    #     train_op_schedule_cpu_general_dx(entities, 5, 16, model_path)
    #     end = time.time()
    #     logs = "{}({}):\n".format("conv3d_channel_batch", args) + "train done! use {}ms\n".format((end - beg) * 1e3)
    #     cost10 = test_graph_schedule_cpu_general_dx(func, args, model_path, number=10)
    #     cost100 = test_graph_schedule_cpu_general_dx(func, args, model_path, number=100)
    #     cost1000 = test_graph_schedule_cpu_general_dx(func, args, model_path, number=1000)
    #     logs = logs + "[10:{},100:{},1000:{}]\n".format(cost10, cost100, cost1000)
    #     with open("logs_conv3d_cpu.txt", "a") as f:
    #         f.write(logs)

    # mttkrp
    # numpy baseline
    # arg_lst = [
    #         (1024, 1024, 8, 8),
    #         (32, 32, 32, 32),
    #         (256, 256, 16, 32)
    #     ]
    #
    #     names = [
    #         "1024_1024_8_8",
    #         "32_32_32_32",
    #         "256_256_16_32"
    #     ]
    #
    #     for i in range(len(arg_lst)):
    #         args = arg_lst[i]
    #         M, N, K, L = args
    #         logs = "{}({}):\n".format("dense_MTTKRP3", args)
    #         cost10 = numpy_baseline(M, N, K, L, 10)
    #         cost100 = numpy_baseline(M, N, K, L, 100)
    #         cost1000 = numpy_baseline(M, N, K, L, 1000)
    #         logs = logs + "[10:{},100:{},1000:{}]\n".format(cost10, cost100, cost1000)
    #         with open("logs_numpy_mttkrp3_cpu.txt", "a") as f:
    #             f.write(logs)
