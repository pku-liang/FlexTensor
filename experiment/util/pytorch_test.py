import torch
import time


def pytorch_cpu(setup_func, stmt_func, number=100):

    setup_func()
    stmt_func()
    run_time = 0.0

    for _ in range(number):
        start = time.time()

        stmt_func()

        end = time.time()
        run_time += end - start

    return run_time / number * 1e3


def pytorch_gpu(setup_func, stmt_func, number=100):

    torch.backends.cudnn.enabled = False
    setup_func()
    stmt_func()
    torch.cuda.synchronize()
    run_time = 0.0

    for _ in range(number):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        stmt_func()

        end.record()
        torch.cuda.synchronize()
        run_time += start.elapsed_time(end)

    return run_time / number

def pytorch_cudnn(setup_func, stmt_func, number=100):

    torch.backends.cudnn.enabled = True
    setup_func()
    stmt_func()
    torch.cuda.synchronize()
    run_time = 0.0

    for _ in range(number):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        stmt_func()

        end.record()
        torch.cuda.synchronize()
        run_time += start.elapsed_time(end)

    return run_time / number


func_dict = {
    "llvm": pytorch_cpu,
    "gpu": pytorch_gpu,
    "cudnn": pytorch_cudnn,
}

def test_pytorch(name, pytorch_func, shapes, dev=0):
    for target in ("llvm", "gpu", "cudnn"):
        for shape in shapes:
            setup_func, stmt_func = pytorch_func(shape, target, dev)
            cost = func_dict[target](setup_func, stmt_func, number=10)
            print("{name}_{shape}_pytorch_{target}: {cost}(ms)".format(
                name=name, shape=shape, target=target, cost=cost))
