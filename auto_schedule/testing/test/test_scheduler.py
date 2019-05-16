import time
import tvm 
from auto_schedule.testing.scheduler import Config
from auto_schedule.testing.task import Task
from auto_schedule.testing.scheduler import schedule, schedule_with_config
from auto_schedule.measure import _evaluate


def test():
    # create an empty task but has the correct key we want
    task = Task("yolo1", None, (1, 3, 448, 448, 64, 7, 2, 3, 1, 1), "llvm", 0)
    beg = time.time()
    # s, bufs, configs = schedule(task.key)
    end = time.time()
    # print(tvm.lower(s, bufs, simple_mode=True))
    # print("######################################")
    # print("op schedules:")
    # for config in configs.op_config_lst:
    #     print("----------------------------------")
    #     for name, value in config.items():
    #         if value:
    #             print(name, value)
    # print("graph schedules:")
    # for name, value in configs.graph_config.items():
    #     if value:
    #         print(name, value)
    op_configs = [
        {
            "spatial": [[1, 1, 1, 1], [1, 1, 1, 3], [454, 1, 1, 1], [1, 227, 2, 1]],
            "unroll": [[1500, 1]]
        },
        {
            "spatial": [[1, 1, 1, 1], [2, 4, 2, 4], [8, 1, 4, 7], [7, 1, 16, 2]],
            "reduce": [[1, 3, 1], [7, 1, 1], [7, 1, 1]],
            "unroll": [[1500, 1]]
        }
    ]
    graph_config = {"inline": [[0, 0]]}
    configs = Config(op_configs, graph_config)

    s, bufs = schedule_with_config(task.key, configs)
    time_cost = _evaluate(s, bufs, "llvm", 0, 10)
    print("Use", time_cost, "ms")
    print("Cost", end - beg, "s")


if __name__ == "__main__":
    test()