import time
import os
import torch
import argparse


def run(N, H, W, C, kernel_size, K, stride, padding, model_path, epoch=5, sample=16, number=100, test=False):
    return -1


def pytorch_baseliine(N, H, W, C, kernel_size, K, stride, padding, number=100, dev=0):
    A = torch.rand([N, C, H, W]).cuda("cuda:" + str(dev))
    W = torch.rand([K, C, kernel_size, kernel_size]).cuda("cuda:" + str(dev))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(number):
        torch.nn.functional.conv2d(A, W)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    return start.elapsed_time(end) / number


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="train the model", action="store_true")
    parser.add_argument("-p", "--pytorch", help="run pytorch baseline", action="store_true")
    parser.add_argument("-a", "--flextensor", help="run auto-scheduler", action="store_true")
    parser.add_argument("-n", "--number", help="number of tests", type=int, default=100)
    parser.add_argument("-f", "--model_file_path", type=str, default="../logs/test_model.pkl")
    parser.add_argument("--params", help="N,H,W,C,k,K,stride,padding", type=str, default="1,14,14,512,3,512,1,1")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--sample", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    test = not args.train
    use_torch = args.pytorch
    use_auto = args.flextensor
    try:
        params = [int(x) for x in args.params.split(",")]
        batch_size, height, width, channel, kernel_size, output_channel, stride, padding = params
        if use_torch:
            cost = pytorch_baseliine(batch_size, height, width, channel, kernel_size, output_channel, stride, padding, args.number, args.device)
            print("PyTorch baseline: {}ms".format(cost))
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
    #     model_path = "opt_conv2d_nchw_" + names[i] + "_gpu.pkl"
    #     entities = []
    #     args = arg_lst[i]
    #     entities.append(Entity("conv2d_nchw", args))
    #     model_path = os.path.abspath(model_path)
    #     train_op_schedule_gpu_general_dx(entities, 40, 25, model_path, logfile="process_conv2d_nchw_" + names[i] + "_gpu.txt")

