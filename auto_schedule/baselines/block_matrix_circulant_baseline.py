import argparse
import time
import copy
import torch
import numpy as np
from auto_schedule.configs.block_circulant_matrix_config import block_circulant_matrix_shapes as shapes

def python_cpu(ROW, COL, FFT, number=10, dev=0):
    Input = np.random.random([ROW, COL]).astype(np.float32)
    Output = np.random.random([ROW, COL]).astype(np.float32)

    def run():
        nonlocal Input, Output
        for i in range(ROW // FFT):
            sub_vec = np.zeros([FFT], dtype=np.float32)
            vec = np.zeros([COL], dtype=np.float32)
            for t in range(COL // FFT):
                for m in range(FFT):
                    for n in range(FFT):
                        vec[t * FFT + m] += \
                            Input[FFT * i + n][t * FFT + (m + n) % FFT] / FFT

            for j in range(FFT):
                for k in range(COL//FFT):
                    if j >= 1:
                        sub_vec[0] = vec[FFT * (k + 1) - 1]
                        sub_vec[1: FFT] = vec[FFT * k: FFT * (k + 1) - 1]
                        vec[FFT * k: FFT * (k + 1)] = sub_vec
                Output[FFT * i + j][:] = copy.deepcopy(vec)

    sum_time = 0.0
    for _ in range(number):
        start = time.time()
        run()
        end = time.time()
        sum_time += end - start
    
    return sum_time / number * 1e3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--from_", help="From which shape", type=int, default=0)
    parser.add_argument("-t", "--to", help="To which shape", type=int, default=-1)
    parser.add_argument("-n", "--number", help="number test run", type=int, default=10)
    parser.add_argument("--target", help="target device type", type=str, default="cpu")
    parser.add_argument("--device", help="target device number", type=int, default=0)
    parser.add_argument("--type", help="type of baseline", type=str, default="python")

    args = parser.parse_args()
    if args.to < 0:
        end = len(shapes)
    else:
        end = args.to
    shapes = shapes[args.from_:end]

    if args.type == "python":
        if args.target == "cpu":
            baseline = python_cpu
        else:
            raise RuntimeError("Only support target 'cpu', but got %s"%args.target)
    else:
        raise RuntimeError("Only implement pytorch baseline now, no '%s' baseline"%args.type)
    
    print("%s baselines convolution 1d for target %s (%d):" % (args.type, args.target, args.device))
    for i, shape in enumerate(shapes):
        count = i + args.from_ 
        print("layer", count)
        cost = baseline(*shape, number=args.number, dev=args.device)
        print("Use %f(ms)" % cost)
    print("Done!")
