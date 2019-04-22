import tvm 
from auto_schedule.testing.ops import conv2d_nchw
from auto_schedule.testing.task import Task, register_task
from auto_schedule.testing.scheduler import schedule, schedule_with_config
from auto_schedule.measure import _evaluate


def conv2d(N, C, H, W, K, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    Img = tvm.placeholder((N, C, H, W))
    W = tvm.placeholder((K, C, kernel_size, kernel_size))
    Output = conv2d_nchw(Img, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return [Output.op], [Img, W, Output]


def test():
    task = Task("yolo24", conv2d, (1, 1024, 7, 7, 1024, 3, 1, 1, 1, 1), "cuda")
    register_task(task)
    s, bufs, configs = schedule(task.key)
    print(tvm.lower(s, bufs, simple_mode=True))
    print("######################################")
    print("op schedules:")
    for config in configs.op_config_lst:
        for name, value in config.items():
            if value:
                print(name, value)
    print("graph schedules:")
    for name, value in configs.graph_config.items():
        if value:
            print(name, value)
    
    s, bufs = schedule_with_config(task.key, configs)
    time_cost = _evaluate(s, bufs, "cuda", 1, 100)
    print("Use", time_cost, "ms")


if __name__ == "__main__":
    test()