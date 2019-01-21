

def base_line(compute, args, target="llvm"):
    """
    base line schedule

    :param compute: function
        a callable object returns list of Operation and list of tvm.Tensor
    :param args:  list | tuple
        arguments for compute
    :param target:  str
        hardware name, e.g. "llvm"
    :return: function
        a callable object produced by tvm.build
    """
    import tvm

    ops, bufs = compute(*args)
    s = tvm.create_schedule(ops)
    func = tvm.build(s, bufs, target)
    return func


def auto_schedule(compute, args, target="llvm", q=None, path="."):
    """
    auto_schedule entry point

    :param compute: function
        a callable object returns list of Operation and list of tvm.Tensor
    :param args:  list | tuple
        arguments for compute
    :param target:  str
        hardware name, e.g. "llvm"
    :param q: multiprocessing.Queue
        result queue
    :param path: str
        where to store result function
    :return: function
        a callable object produced by tvm.build
    """
    import os
    import tvm
    from .Scheduler import OldTrainer
    from .Evaluator import TVMEvaluator

    ops, bufs = compute(*args)
    trainer = OldTrainer(ops, bufs, target)
    evaluator = TVMEvaluator(compute, args, ops, bufs)
    func_name, cost = trainer.start(evaluator, epoch=2, path=path)
    if q:
        q.put((func_name, cost))
    return tvm.module.load(os.path.join(path, func_name+".so"))


def multi_auto_schedule(compute, args, target="llvm", p_num=2):
    import os
    import tvm
    import multiprocessing as multi

    print("Starting multiprocess auto-scheduler...", flush=True)
    path = "./funcs"
    # delete previous results
    if os.path.isdir(path):
        os.rmdir(path)
    os.makedirs(path)
    processes = []
    q = multi.Queue()
    for i in range(p_num):
        processes.append(multi.Process(target=auto_schedule, args=(compute, args, target, q, path)))
    for process in processes:
        process.start()

    best_func = None
    least_cost = float("+inf")
    count = 0
    while count < p_num:
        func_name, cost = q.get()
        count += 1
        func = tvm.module.load(os.path.join(path, func_name+".so"))
        if cost < least_cost:
            best_func = func
            least_cost = cost
    for process in processes:
        process.terminate()
    print("All done! Exit...", flush=True)
    return best_func
