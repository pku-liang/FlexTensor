import tvm
import logging
import multiprocessing as multi
from auto_schedule.scheduler import evaluate
from auto_schedule.training_examples import FUNC_TABLE


CPU_DEV_ID = 0
CUDA_DEV_ID = 0
VERBOSE = False
REPORT = True


def cuda_gemv(args):
    func = FUNC_TABLE["gemv"].func
    ops, bufs = func(*args)
    Cop = ops
    s = tvm.create_schedule(Cop)
    bx = tvm.thread_axis("blockIdx.x")
    bf = 1
    h = Cop.axis[0]
    ho, hi = s[Cop].split(h, nparts=bf)
    s[Cop].bind(ho, bx)
    if VERBOSE:
        print(tvm.lower(s, bufs, simple_mode=True))
    time_cost = evaluate(s, bufs, "cuda", CUDA_DEV_ID)
    return time_cost


def cuda_matmul(args):
    func = FUNC_TABLE["matmul"].func
    ops, bufs = func(*args)
    Cop = ops
    s = tvm.create_schedule(Cop)
    bx = tvm.thread_axis("blockIdx.x")
    tx = tvm.thread_axis("threadIdx.x")
    bf = 1
    tf = 1
    h, w = Cop.axis
    ho, hi = s[Cop].split(h, nparts=bf)
    wo, wi = s[Cop].split(w, nparts=tf)
    s[Cop].bind(ho, bx)
    s[Cop].bind(wo, tx)
    if VERBOSE:
        print(tvm.lower(s, bufs, simple_mode=True))
    time_cost = evaluate(s, bufs, "cuda", CUDA_DEV_ID)
    return time_cost


def cuda_matmul_batch(args):
    func = FUNC_TABLE["matmul_batch"].func
    ops, bufs = func(*args)
    Cop = ops
    s = tvm.create_schedule(Cop)
    bx = tvm.thread_axis("blockIdx.x")
    tx = tvm.thread_axis("threadIdx.x")
    bf = 1
    tf = 1
    b, h, w = Cop.axis
    bo, bi = s[Cop].split(b, nparts=bf)
    ho, hi = s[Cop].split(h, nparts=tf)
    s[Cop].bind(bo, bx)
    s[Cop].bind(ho, tx)
    if VERBOSE:
        print(tvm.lower(s, bufs, simple_mode=True))
    time_cost = evaluate(s, bufs, "cuda", CUDA_DEV_ID)
    return time_cost


def cuda_conv2d_channel_batch(args):
    func = FUNC_TABLE["conv2d_channel_batch"].func
    ops, bufs = func(*args)
    Output = ops
    Apad = ops.input_tensors[0].op
    s = tvm.create_schedule(Output)
    bx = tvm.thread_axis("blockIdx.x")
    tx = tvm.thread_axis("threadIdx.x")
    bf = 1
    tf = 1
    b, h, w, c = Output.axis
    bo, bi = s[Output].split(b, nparts=bf)
    ho, hi = s[Output].split(h, nparts=tf)
    s[Output].bind(bo, bx)
    s[Output].bind(ho, tx)
    b, h, w, c = Apad.axis
    bo, bi = s[Apad].split(b, nparts=bf)
    ho, hi = s[Apad].split(h, nparts=tf)
    s[Apad].bind(bo, bx)
    s[Apad].bind(ho, tx)
    if VERBOSE:
        print(tvm.lower(s, bufs, simple_mode=True))
    time_cost = evaluate(s, bufs, "cuda", CUDA_DEV_ID)
    return time_cost


def cpu_gemv(args):
    func = FUNC_TABLE["gemv"].func
    ops, bufs = func(*args)
    Cop = ops
    s = tvm.create_schedule(Cop)
    if VERBOSE:
        print(tvm.lower(s, bufs, simple_mode=True))
    time_cost = evaluate(s, bufs, "llvm", CPU_DEV_ID)
    return time_cost


def cpu_matmul(args):
    func = FUNC_TABLE["matmul"].func
    ops, bufs = func(*args)
    Cop = ops
    s = tvm.create_schedule(Cop)
    if VERBOSE:
        print(tvm.lower(s, bufs, simple_mode=True))
    time_cost = evaluate(s, bufs, "llvm", CPU_DEV_ID)
    return time_cost


def cpu_matmul_batch(args):
    func = FUNC_TABLE["matmul_batch"].func
    ops, bufs = func(*args)
    Cop = ops
    s = tvm.create_schedule(Cop)
    if VERBOSE:
        print(tvm.lower(s, bufs, simple_mode=True))
    time_cost = evaluate(s, bufs, "llvm", CPU_DEV_ID)
    return time_cost


def cpu_conv2d_channel_batch(args):
    func = FUNC_TABLE["conv2d_channel_batch"].func
    ops, bufs = func(*args)
    Cop = ops
    s = tvm.create_schedule(Cop)
    if VERBOSE:
        print(tvm.lower(s, bufs, simple_mode=True))
    time_cost = evaluate(s, bufs, "llvm", CPU_DEV_ID)
    return time_cost


def generate():
    # matmul
    # cuda_gemv_p = multi.Pool(5)
    cpu_gemv_p = multi.Pool(5)
    gemv_name_lst = []
    gemv_args_lst = []
    # cuda_matmul_p = multi.Pool(5)
    cpu_matmul_p = multi.Pool(5)
    matmul_name_lst = []
    matmul_args_lst = []
    # cuda_matmul_batch_p = multi.Pool(5)
    cpu_matmul_batch_p = multi.Pool(5)
    matmul_batch_name_lst = []
    matmul_batch_args_lst = []
    for s in [64, 128, 200, 256, 400, 512, 1024, 2048, 4096]:
        gemv_name_lst.append("gemv_{}".format(s))
        gemv_args_lst.append((s, s))
        matmul_name_lst.append("matmul_{}".format(s))
        matmul_args_lst.append((s, s, s))
        matmul_batch_name_lst.append("matmul_batch_{}".format(s))
        matmul_batch_args_lst.append((256, s, s, s))

    # test
    # gemv_name_lst = []
    # gemv_args_lst = []
    # matmul_name_lst = []
    # matmul_args_lst = []
    # matmul_batch_name_lst = []
    # matmul_batch_args_lst = []
    # for s in range(10):
    #     gemv_name_lst.append("gemv_test_{}".format(s))
    #     gemv_args_lst.append((10 + s, 10 + s))
    #     matmul_name_lst.append("matmul_test_{}".format(s))
    #     matmul_args_lst.append((10 + s, 10 + s, 10 + s))
    #     matmul_batch_name_lst.append("matmul_batch_test_{}".format(s))
    #     matmul_batch_args_lst.append((256, 10 + s, 10 + s, 10 + s))

    # cuda_gemv_time_lst = cuda_gemv_p.map(cuda_gemv, gemv_args_lst)
    # cuda_matmul_time_lst = cuda_matmul_p.map(cuda_matmul, matmul_args_lst)
    # cuda_matmul_batch_time_lst = cuda_matmul_batch_p.map(cuda_matmul_batch, matmul_batch_args_lst)

    cpu_gemv_time_lst = cpu_gemv_p.map(cpu_gemv, gemv_args_lst)
    cpu_matmul_time_lst = cpu_matmul_p.map(cpu_matmul, matmul_args_lst)
    cpu_matmul_batch_time_lst = cpu_matmul_batch_p.map(cpu_matmul_batch, matmul_batch_args_lst)

    conv_name_lst = [
        "YOLO_conv5",
        "YOLO_conv6",
        "YOLO_conv7",
        "YOLO_conv8",
        "YOLO_conv10",
        "YOLO_conv15",
        "YOLO_conv16",
        "YOLO_conv17",
        "YOLO_conv18",
        "YOLO_conv21",
        "YOLO_conv22",
        "YOLO_conv23"
    ]
    conv_args_lst = [
        (256, 56, 56, 256, 1, 1, 256, 1, 0),
        (256, 56, 56, 256, 3, 3, 512, 1, 1),
        (256, 28, 28, 512, 1, 1, 256, 1, 0),
        (256, 28, 28, 256, 1, 1, 512, 1, 0),
        (256, 28, 28, 256, 3, 3, 512, 1, 1),
        (256, 28, 28, 512, 1, 1, 512, 1, 0),
        (256, 28, 28, 512, 3, 3, 1024, 1, 1),
        (256, 14, 14, 1024, 1, 1, 512, 1, 0),
        (256, 14, 14, 512, 3, 3, 1024, 1, 1),
        (256, 14, 14, 1024, 3, 3, 1024, 1, 1),
        (256, 14, 14, 1024, 3, 3, 1024, 2, 1),
        (256, 7, 7, 1024, 3, 3, 1024, 1, 1)
    ]

    # test
    # conv_name_lst = []
    # conv_args_lst = []
    # for s in range(10):
    #     conv_name_lst.append("conv_test_{}".format(s))
    #     conv_args_lst.append((256, 10 + s, 10 + s, 3, 1, 1, 3, 1, 0))

    # cuda_conv_p = multi.Pool(5)
    cpu_conv_p = multi.Pool(5)

    # cuda_conv_time_lst = cuda_conv_p.map(cuda_conv2d_channel_batch, conv_args_lst)
    cpu_conv_time_lst = cpu_conv_p.map(cpu_conv2d_channel_batch, conv_args_lst)
    
    # pools = [cuda_gemv_p, cuda_matmul_p, cuda_matmul_batch_p, cuda_conv_p]
    # for pool in pools:
    #     pool.close()
    #     pool.join()

    # with open("cuda_baseline.txt", 'w') as f:
    #     for i, name in enumerate(gemv_name_lst):
    #         line = "{} {}\n".format(name, cuda_gemv_time_lst[i])
    #         f.write(line)
    #     for i, name in enumerate(matmul_name_lst):
    #         line = "{} {}\n".format(name, cuda_matmul_time_lst[i])
    #         f.write(line)
    #     for i, name in enumerate(matmul_batch_name_lst):
    #         line = "{} {}\n".format(name, cuda_matmul_batch_time_lst[i])
    #         f.write(line)
    #     for i, name in enumerate(conv_name_lst):
    #         line = "{} {}\n".format(name, cuda_conv_time_lst[i])
    #         f.write(line)

    pools = [cpu_gemv_p, cpu_matmul_p, cpu_matmul_batch_p, cpu_conv_p]
    for pool in pools:
        pool.close()
        pool.join()
        
    with open("cpu_baseline_test.txt", 'w') as f:
        for i, name in enumerate(gemv_name_lst):
            line = "{} {}\n".format(name, cpu_gemv_time_lst[i])
            f.write(line)
        for i, name in enumerate(matmul_name_lst):
            line = "{} {}\n".format(name, cpu_matmul_time_lst[i])
            f.write(line)
        for i, name in enumerate(matmul_batch_name_lst):
            line = "{} {}\n".format(name, cpu_matmul_batch_time_lst[i])
            f.write(line)
        for i, name in enumerate(conv_name_lst):
            line = "{} {}\n".format(name, cpu_conv_time_lst[i])
            f.write(line)


if __name__ == "__main__":
    logger = logging.getLogger("generate")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename="generate_test.log")
    logger.addHandler(fh)
    logger.info("generate begins...")
    generate()
    logger.info("generate done!")