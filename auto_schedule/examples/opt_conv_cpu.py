import time
import os
import torch
import argparse
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.test import test_graph_schedule_cpu_general_dx
from auto_schedule.train import Entity, train_op_schedule_cpu_general_dx


def run(batch_size, height, width, channel, kernel_size, output_channel, stride, padding,
        model_path, epoch=5, sample_size=16, number=100, test=False):
    entities = []
    func = FUNC_TABLE["conv2d_channel_batch"].func
    args = (batch_size, height, width, channel, kernel_size, kernel_size, output_channel, stride, padding)
    entities.append(Entity("conv2d_channel_batch", args))
    model_path = os.path.abspath(model_path)
    if not test:
        beg = time.time()
        train_op_schedule_cpu_general_dx(entities, epoch, sample_size, model_path)
        end = time.time()
        print("{}({}):".format("conv2d_channel_batch", args))
        print("train done! use {}ms".format((end - beg) * 1e3))
    test_graph_schedule_cpu_general_dx(func, args, model_path, number=number)


def pytorch_baseliine(batch_size, height, width, channel, kernel_size, output_channel, stride, padding, number=100):
    conv = torch.nn.Conv2d(channel, output_channel, (kernel_size, kernel_size), (stride, stride), (padding, padding), bias=False).cuda()
    A = torch.rand([batch_size, channel, height, width]).cuda()
    beg = time.time()
    for i in range(number):
        conv(A)
    end = time.time()
    print("pytorch use {}ms".format((end - beg) / number * 1e3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="train the model", action="store_true")
    parser.add_argument("-p", "--pytorch", help="run pytorch baseline", action="store_true")
    parser.add_argument("-a", "--auto_schedule", help="run auto-scheduler", action="store_true")
    parser.add_argument("-n", "--number", help="number of tests", type=int, default=100)
    parser.add_argument("-f", "--model_file_path", type=str, default="../logs/test_model.pkl")
    parser.add_argument("--params", help="N,H,W,C,k,K,stride,padding", type=str, default="1,14,14,512,3,512,1,1")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--sample", type=int, default=16)
    args = parser.parse_args()
    test = not args.train
    use_torch = args.pytorch
    use_auto = args.auto_schedule
    try:
        params = [int(x) for x in args.params.split(",")]
        batch_size, height, width, channel, kernel_size, output_channel, stride, padding = params
        if use_torch:
            pytorch_baseliine(batch_size, height, width, channel, kernel_size, output_channel, stride, padding, args.number)
        if use_auto:
            run(batch_size, height, width, channel, kernel_size, output_channel, stride, padding,
                args.model_file_path, args.epoch, args.sample, args.number, test)
    except Exception as e:
        raise ValueError("Bad parameters, please refer to usage")
    # arg_lst = [
    #     (1, 7, 7, 1024, 3, 3, 1024, 1, 1),
    #     # (8, 7, 7, 1024, 3, 3, 1024, 1, 1),
    #     # (64, 7, 7, 1024, 3, 3, 1024, 1, 1),
    #     # (256, 7, 7, 1024, 3, 3, 1024, 1, 1),
    #     (1, 14, 14, 1024, 1, 1, 512, 1, 0),
    #     (1, 28, 28, 256, 3, 3, 512, 1, 1),
    #     (1, 28, 28, 512, 1, 1, 256, 1, 0),
    #     (1, 56, 56, 128, 3, 3, 256, 1, 1),
    #     (1, 56, 56, 192, 1, 1, 128, 1, 0),
    #     (1, 112, 112, 64, 3, 3, 192, 1, 1),
    #     (1, 448, 448, 3, 7, 7, 64, 2, 3)
    # ]
    # names = [
    #     "yolo24_b1",
    #     # "yolo24_b8",
    #     # "yolo24_b64",
    #     # "yolo24_b256",
    #     "yolo19_b1",
    #     "yolo10_b1",
    #     "yolo7_b1",
    #     "yolo4_b1",
    #     "yolo3_b1",
    #     "yolo2_b1",
    #     "yolo1_b1"
    # ]
    # for i in range(len(arg_lst)):
    #     # model_path = "opt_conv2d_nchw_" + names[i] + "_cpu.pkl"
    #     # entities = []
    #     # args = arg_lst[i]
    #     # entities.append(Entity("conv2d_nchw", args))
    #     # model_path = os.path.abspath(model_path)
    #     # train_op_schedule_cpu_general_dx(entities, 20, 50, model_path,
    #     #                                  logfile="process_conv2d_nchw_" + names[i] + "_cpu.txt", device="cuda:2")
    #     N, H, W, CI, k, _, CO, S, P = arg_lst[i]
    #     print(names[i], pytorch_baseliine(N, H, W, CI, k, CO, S, P, 10))