import torch
import tvm

# Currently not used
index2action = []
action2index = {}


ACTION_TABLE = {}


def register(action_class):
    ACTION_TABLE[action_class.__name__] = action_class


class BaseAction(object):
    """
    Action in reinforcement learning
    Represents for different schedule primitives
    """
    def apply(self, s, env, node, arg_bufs, target, fast_mode=False):
        """
        Try to carry out the schedule primitive
        to the given tvm.Schedule object in a simple way

        :param s:  tvm.Schedule
        :param node:  oNode | subclass of  oNode
        :param arg_bufs: list of tvm.Tensor
            used in tvm.build
        :param target: str
            used in tvm.build
        :return: bool, str
            whether carry out successfully
            message of errors
        """
        raise NotImplementedError("Should use subclass of Action to call apply")

    def name(self):
        return self.__class__.__name__


class ComputeAt(BaseAction):
    """
    Action for compute_at primitive
    """
    def __init__(self, axis_name):
        super(ComputeAt, self).__init__()
        self._axis_name = axis_name

    def apply(self, s, env, node, arg_bufs, target, fast_mode=False):
        good = True
        msg = ""

        next_node = env.op2node[env.down_graph[node.op][0]]

        axis = next_node.get_real_axis(self._axis_name)  # IterVar
        # Try to compute_at
        try:
            s[node.op].compute_at(s[next_node.op], axis)
            if not fast_mode:
                tvm.build(s, arg_bufs, target)
        except Exception as e:
            good = False
            msg = str(e)

        return good, msg

    def to_json(self):
        ret = dict()
        ret["type"] = self.name()
        ret["axis_name"] = self._axis_name
        return ret

    @classmethod
    def from_json(cls, sch_json):
        if "axis_name" not in sch_json:
            raise ValueError("can't convert to ComputeAt Action: [", sch_json, "]")
        return cls(sch_json["axis_name"])

    def __str__(self):
        return "compute_at(axis={})".format(self._axis_name)


class ComputeInline(BaseAction):
    """
    Action for compute_inline primitive
    """
    def __init__(self):
        super(ComputeInline, self).__init__()

    def apply(self, s, env, node, arg_bufs, target, fast_mode=False):
        good = True
        msg = ""

        # Try to compute_inline
        try:
            s[node.op].compute_inline()
            if not fast_mode:
                tvm.build(s, arg_bufs, target)
        except Exception as e:
            good = False
            msg = str(e)
        if good:
            node.isolate = False
        return good, msg

    def to_json(self):
        ret = dict()
        ret["type"] = self.name()
        return ret

    @classmethod
    def from_json(cls, sch_json):
        return cls()

    def __str__(self):
        return "compute_inline()"


class Split(BaseAction):
    """
    Action for split primitive
    """
    def __init__(self, axis_name, split_factors):
        """
        :param axis: str
            the name of axis to split
        :param nparts: int
            number of parts to split
        """
        super(Split, self).__init__()
        self._axis_name = axis_name
        self._split_factors = split_factors
        self._result_axis_names = None

    def get_result_axis_names(self):
        return self._result_axis_names

    def apply(self, s, env, node, arg_bufs, target, fast_mode=False):
        good = True
        msg = ""
        # use these to update node axis features
        new_axes = []

        axis = node.get_real_axis(self._axis_name)   # IterVar
        # Try to split
        try:
            for factor in self._split_factors[:-1]:
                axis, tmp = s[node.op].split(axis, factor=factor)
                new_axes.append(tmp)
            new_axes.append(axis)
            if not fast_mode:
                tvm.build(s, arg_bufs, target)
        except Exception as e:
            good = False
            msg = str(e)

        if good:
            self._result_axis_names = node.update_real_axes(self._axis_name, list(reversed(new_axes)))
        return good, msg

    def to_json(self):
        ret = dict()
        ret["type"] = self.name()
        ret["axis_name"] = self._axis_name
        ret["all_factors"] = list(self._split_factors)
        return ret

    @ classmethod
    def from_json(cls, sch_json):
        if ("axis_name" not in sch_json) or ("all_factors" not in sch_json):
            raise ValueError("can't convert to Split Action: [", sch_json, "]")
        return cls(sch_json["axis_name"], sch_json["all_factors"])

    def __str__(self):
        return "split({}, factors={})".format(self._axis_name, list(reversed(self._split_factors)))


class Reorder(BaseAction):
    """
    Action for reorder primitive
    """
    def __init__(self, axis_names):
        super(Reorder, self).__init__()
        self._axis_names = axis_names

    def apply(self, s, env, node, arg_bufs, target, fast_mode=False):
        good = True
        msg = ""
        real_axes = []
        for axis in self._axis_names:
            real_axes.append(node.get_real_axis(axis))

        # Try to reorder
        try:
            s[node.op].reorder(*real_axes)
            if not fast_mode:
                tvm.build(s, arg_bufs, target)
        except Exception as e:
            good = False
            msg = str(e)

        # If everything goes well
        if good:
            node.update_axes_order(self._axis_names)
        return good, msg

    def to_json(self):
        ret = dict()
        ret["type"] = self.name()
        ret["axis_names"] = list(self._axis_names)
        return ret

    @classmethod
    def from_json(cls, sch_json):
        if "axis_names" not in sch_json:
            raise ValueError("can't convert to Reorder: [", sch_json, "]")
        return cls(sch_json["axis_names"])

    def __str__(self):
        record_str = "reorder("
        record_str += self._axis_names[0]
        for i in range(1, len(self._axis_names)):
            record_str += ", "
            record_str += self._axis_names[i]
        record_str += ")"
        return record_str


class Fuse(BaseAction):
    """
    Action for fuse primitive
    """
    def __init__(self, axis_list):
        super(Fuse, self).__init__()
        self._axis_list = axis_list

    def apply(self, s, env, node, arg_bufs, target, fast_mode=False):
        pass

    def template_apply(self, s, record, cfg, record_mode=False, records=None, others=None):
        pass


class Parallel(BaseAction):
    """
    Action for parallel primitive
    """
    def __init__(self, axis):
        super(Parallel, self).__init__()
        self._axis = axis

    def apply(self, s, env, node, arg_bufs, target, fast_mode=False):
        pass

    def template_apply(self, s, record, cfg, record_mode=False, records=None, others=None):
        pass


class Vectorize(BaseAction):
    """
    Action for vectorize primitive
    """
    def __init__(self, axis):
        super(Vectorize, self).__init__()
        self._axis = axis

    def apply(self, s, env, node, arg_bufs, target, fast_mode=False):
        pass

    def template_apply(self, s, record, cfg, record_mode=False, records=None, others=None):
        pass


class Unroll(BaseAction):
    """
    Action for unroll primitive
    """
    def __init__(self, axis):
        super(Unroll, self).__init__()
        self._axis = axis

    def apply(self, s, env, node, arg_bufs, target, fast_mode=False):
        pass

    def template_apply(self, s, record, cfg, record_mode=False, records=None, others=None):
        pass


# Currently not used
def load_actions_embedding(env, f=None):
    embedding = None
    if f is None:
        embedding = torch.eye(len(env.index2action))
    else:
        pass  # TODO load pretrained embeddings
    return embedding


# Currently not used
def load_actions():
    index2action.extend([ComputeAt, ComputeInline, Split, Reorder, Fuse, Parallel, Vectorize, Unroll])
    for i, action in enumerate(index2action):
        action2index[action] = i


register(Split)
register(Reorder)
register(ComputeAt)
register(ComputeInline)
register(Fuse)
register(Unroll)
register(Parallel)
register(Vectorize)
