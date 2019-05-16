import argparse
import timeit
import torch
from auto_schedule.testing.configs.gemv_config import gemv_shapes
torch.backends.cudnn.enabled = False


def pytorch_cpu(N, K, number=100, dev=0):
    run_time = timeit.timeit(setup= 'import torch\n'
                                    'A = torch.rand([' + str(N) + ', ' + str(K) + '], dtype=torch.float32)\n'
                                    'B = torch.rand([' + str(K) + ', 1], dtype=torch.float32)\n'
                                    'torch.mm(A, B)\n',
                               stmt='ans = torch.mm(A, B)',
                               number=number)
    return run_time / number * 1e3


def pytorch_cuda(N, K, number=100, dev=0):
    A = torch.rand([N, K], dtype=torch.float32).cuda("cuda:" + str(dev))
    B = torch.rand([K, 1], dtype=torch.float32).cuda("cuda:" + str(dev))

    # warm-up
    torch.mm(A, B)
    torch.cuda.synchronize()
    sum_time = 0.0
    for i in range(number):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ans = torch.mm(A, B)
        end.record()
        
        # Waits for everything to finish running
        torch.cuda.synchronize()
        sum_time += start.elapsed_time(end)

    return sum_time / number


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--from_", help="From which shape", type=int, default=0)
    parser.add_argument("-t", "--to", help="To which shape", type=int, default=-1)
    parser.add_argument("-n", "--number", help="number test run", type=int, default=10)
    parser.add_argument("--target", help="target device type", type=str, default="llvm")
    parser.add_argument("--device", help="target device number", type=int, default=0)
    parser.add_argument("--type", help="type of baseline", type=str, default="pytorch")

    args = parser.parse_args()
    shapes = gemv_shapes
    if args.to < 0:
        end = len(shapes)
    else:
        end = args.to
    shapes = shapes[args.from_:end]
    if args.type == "pytorch":
        if args.target == "cuda":
            baseline = pytorch_cuda
        elif args.target == "llvm":
            baseline = pytorch_cpu
        else:
            raise RuntimeError("Only support target 'llvm' and 'cuda', but got %s"%args.target)
    else:
        raise RuntimeError("Only implement pytorch baseline now, no '%s' baseline"%args.type)
    
    print("%s baselines gemv for target %s (%d):" % (args.type, args.target, args.device))
    for i, shape in enumerate(shapes):
        count = i + args.from_ 
        print("layer", count)
        N, K, _ = shape
        cost = baseline(N, K, args.number, args.device)
        print("Use %f(ms)" % cost)
    print("Done!")
