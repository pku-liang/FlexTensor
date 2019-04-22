TASK_TABLE = {}


class Task(object):
    def __init__(self, name, func, args, target):
        self.key = "{}_{}_{}".format(name, args, target)
        self.func = func
        self.args = args
        self.target = target


def register_task(task, override=False):
    if task.key in TASK_TABLE and not override:
        raise RuntimeError("Same task occurs again")
    TASK_TABLE[task.key] = task