from __future__ import absolute_import

from auto_schedule.task import TASK_TABLE
from get_relation import GetRelationOp


def test_get_relation():
    for key, task in TASK_TABLE.items():
        if "gemm_conv2d" in key or "shift" in key:
            continue
        print("TASK Name:", key)
        ops, bufs = task.func(*task.args)
        visitor = GetRelationOp(ops[0])
        visitor.get_relation()
        for res in visitor.res_lst:
            print(str(res))
        print("*************************")


if __name__ == "__main__":
    test_get_relation()