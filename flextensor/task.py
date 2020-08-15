import tvm 

from flextensor.space import EnumSpace

TASK_TABLE = {}


class Task(object):
    def __init__(self, category, name, func, args, target, dev_id=0):
        self.key = "{}_{}_{}_{}({})".format(category, name, args, target, dev_id)#
        self.func = func
        self.args = args
        self.target = target
        self.dev_id = dev_id
        self.category = category
        self.special_space = {}

    def set_specific_space(self, key, type, knobs):
        if type == "enum":
            self.special_space[key] = EnumSpace(knobs)
        else:
            raise RuntimeError("Not supported space type: %s" % type)


def register_task(task, override=False):
    if task.key in TASK_TABLE and not override:
        pass # print("[Warning]: Same task occurs again %s" % task.key)
    TASK_TABLE[task.key] = task


def register(func, category, name, args, target, dev_id=0, override=False):
    task = Task(category, name, func, args, target, dev_id)
    register_task(task, override=override)
