"""
Environment for reinforcement learning

Actually, the environment is a graph of nodes, constructed from compute definition.
Overview:
    There are mainly two categories of class: Env and Node
    Other functions are used to construct Nodes and Env
"""
import copy
import tvm
from tvm.autotvm import util
from collections import deque, namedtuple

# A number used when not sure of the shape of certain dimension
NotSure = 512


class AxisFeature(object):
    def __init__(self, pos, dom, a, b):
        self.pos = pos
        self.dom = dom
        self.a = a
        self.b = b

    def __iter__(self):
        return [self.pos, self.dom, self.a, self.b].__iter__()

    def __str__(self):
        return "AxisFeature(pos={}, dom={}, a={}, b={})" \
            .format(self.pos, self.dom, self.a, self.b)


class AxisState(object):
    def __init__(self, reduce, parallel, unroll, vectorize):
        self.reduce = reduce
        self.parallel = parallel
        self.unroll = unroll
        self.vectorize = vectorize

    def __iter__(self):
        return [self.reduce, self.parallel, self.unroll, self.vectorize].__iter__()

    def tolist(self):
        return [self.reduce, self.parallel, self.unroll, self.vectorize]

    def __str__(self):
        return "AxisState(reduce={}, parallel={}, unroll={}, vectorize={})"\
            .format(self.reduce, self.parallel, self.unroll, self.vectorize)


def reduce_axis_state():
    return AxisState(1, 0, 0, 0)


def normal_axis_state():
    return AxisState(0, 0, 0, 0)


class ComputeGraph(object):
    """
    class Env is contains all the useful information of a compute definition,
    and it is used for agents in reinforcement learning process
    """

    def __init__(self, ops):
        """
        :param ops: list of Operation | Operation
            the output Operations of a compute definition
        """
        if not isinstance(ops, (list, tuple)):
            ops = [ops]
        self.op2node = {}
        self.op2index = {}  # map from Operation to the index, used with index2op
        self.index2op = []  # map from index to Operation, used with op2index
        self.src = []  # source Operations of a compute graph
        self.end = []  # destination Operations of a compute graph
        self.down_graph = {}  # map from Operation to list of Operation, i.e. op->output_ops
        self.entry(ops)  # use this to initialize Env

    def entry(self, ops):
        """
        The entry point of initializing a new Env object

        :param env: Env
            Env object to initialize
        :param ops: list of Operation | Operation
        :return: None
        """
        # Initialize src, end, index2op
        bfs(self, ops)
        # Initialize op2node, op2index
        generate_nodes(self)

    def print_(self):
        for op, node in self.op2node.items():
            node.print_()

    def reset(self):
        for op, node in self.op2node.items():
            if not isinstance(node, ComputeVertex):
                continue
            node.reset_all()


def generate_nodes(env):
    """
    Make nodes for operations

    :param env:  Env
        Environment in which these nodes reside
    :return:  None
    """
    broad_order = env.index2op
    for i, op in enumerate(broad_order):
        if isinstance(op, tvm.tensor.PlaceholderOp):
            env.op2node[op] = PlaceholderVertex(env, op)
        elif isinstance(op, tvm.tensor.ComputeOp):
            env.op2node[op] = ComputeVertex(env, op)
        env.op2index[op] = i


def bfs(env, ops):
    """
    Broad order first traverse the compute graph
    record the order, the output operations, sources and ends

    :param env: Env
        Environment in which operations are in
    :param ops: list of Operation
    :return:  None
    """
    reverse_broad_order = []
    srcs = []

    q = deque()
    visit = set()
    for op in ops:
        q.append(op)
        visit.add(op)
    while q:
        op = q.popleft()
        reverse_broad_order.append(op)
        for t in op.input_tensors:
            if t.op not in visit:
                q.append(t.op)
                visit.add(t.op)
            if t.op not in env.down_graph:
                env.down_graph[t.op] = []
            env.down_graph[t.op].append(op)
        if not op.input_tensors:
            srcs.append(op)

    broad_order = list(reversed(reverse_broad_order))
    env.index2op = broad_order
    env.end = ops
    env.src = srcs


class OperationVertex(object):
    """
    class OperationVertex is the base class of different kinds of vertexes
    """

    def __init__(self, env, op):
        self._env = env
        self.op = op
        # Information for schedule
        self._schedule_record = []

        try:    # get the shape of output tensor
            self._shape = util.get_const_tuple(op.output(0).shape)
        except Exception:   # if failed, try estimate a shape
            tmp_shape = []
            for l in op.output(0).shape:
                try:
                    tmp = int(l)
                except Exception:
                    tmp = NotSure
                tmp_shape.append(tmp)
            self._shape = tuple(tmp_shape)

    def print_(self):
        print("-------------------------------------------------")
        print("Operation Vertex {} is wrapper for operation {}:".format(self, self.op))
        print("    output shape: {}".format(self._shape))
        print("-------------------------------------------------")

    def get_shape(self):
        return self._shape

    def add_schedule(self, sch):
        """
        Put into a new schedule primitive

        :param sch: Action
            represents the schedule primitive, also actions in reinforcement learning
        :return: None
        """
        self._schedule_record.append(sch)

    def extend_schedule(self, sch_list):
        self._schedule_record.extend(sch_list)

    def set_schedule(self, sch_list):
        self._schedule_record = sch_list

    def pop_schedule(self):
        return self._schedule_record.pop()

    def get_schedule_list(self):
        """
        Return all the schedule primitives

        :return: list of Action
        """
        return self._schedule_record

    def clear_schedule_list(self):
        """
        Clear schedule list

        :return: None
        """
        self._schedule_record.clear()


class PlaceholderVertex(OperationVertex):
    def __init__(self, env, op):
        super(PlaceholderVertex, self).__init__(env, op)


class ComputeVertex(OperationVertex):
    def __init__(self, env, op):
        super(ComputeVertex, self).__init__(env, op)

        self.isolate = True
        self.call_times = 1

        self.has_reduce = bool(op.reduce_axis)

        # Information for index axis and feature
        self._axis2index = {}
        self._index2axis = []
        self._axis2state = {}    # axis_name->AxisState
        self._axis2dom = {}
        self._axis2real = {}  # axis_name->IterVar

        # axis information
        counter = 0
        for axis in op.axis:
            self._index2axis.append(axis.var.name)
            self._axis2real[axis.var.name] = axis
            self._axis2state[axis.var.name] = normal_axis_state()
            try:
                tmp = int(tvm.ir_pass.Simplify(axis.dom.extent))
            except Exception:
                tmp = NotSure
            self._axis2dom[axis.var.name] = tmp
            self._axis2index[axis.var.name] = counter
            counter += 1

        for axis in op.reduce_axis:
            self._index2axis.append(axis.var.name)
            self._axis2real[axis.var.name] = axis
            self._axis2state[axis.var.name] = reduce_axis_state()
            try:
                tmp = int(tvm.ir_pass.Simplify(axis.dom.extent))
            except Exception:
                tmp = NotSure
            self._axis2dom[axis.var.name] = tmp
            self._axis2index[axis.var.name] = counter
            counter += 1

        self._axis2feature = visit_feature(self._env, self.op, self._index2axis, self._axis2dom).features

        # caches that changes while scheduling
        self._cache_axis2index = {}
        self._cache_index2axis = []
        self._cache_axis2dom = {}
        self._cache_axis2feature = {}
        self._cache_axis2state = {}
        self._cache_axis2real = {}

        self.reset_all()

    def reset_all(self):
        self.isolate = True
        self._cache_axis2index = self._axis2index.copy()
        self._cache_index2axis = self._index2axis.copy()
        self._cache_axis2dom = self._axis2dom.copy()
        self._cache_axis2feature = copy.deepcopy(self._axis2feature)
        self._cache_axis2state = copy.deepcopy(self._axis2state)
        # this is confusing, we should use deepcopy here, but deepcopy doesn't work for tvm IterVar
        # however, shallow copy works well, no errors, this may be because these IterVars aren't changed
        # while scheduling
        self._cache_axis2real = self._axis2real.copy()

    def reset_real_axes(self):
        self._cache_axis2real = self._axis2real.copy()

    def get_copy(self):
        return self._axis2index.copy(), self._index2axis.copy(), self._axis2dom.copy(), \
               copy.deepcopy(self._axis2feature), copy.deepcopy(self._axis2state), self._axis2real.copy()

    def get_pos(self, var_name):
        return self._cache_axis2index[var_name]

    def get_var_names(self):
        return self._cache_index2axis

    def get_original_var_names(self):
        return self._index2axis

    def get_axis_dom(self, var_name):
        return self._cache_axis2dom[var_name]

    def get_real_axis(self, axis_name):
        return self._cache_axis2real[axis_name]

    def get_axis_feature(self, var_name):
        return self._cache_axis2feature[var_name]

    def get_axis_features(self):
        return self._cache_axis2feature

    def get_axis_state(self, var_name):
        return self._cache_axis2state[var_name]

    def print_(self):
        super(ComputeVertex, self).print_()
        print("*************************************************")
        print("    ComputeVertex has axes:")
        for i, var_name in enumerate(self._index2axis):
            print("    axis {}:".format(var_name))
            print("    state: {}".format(self._axis2state[var_name]))
            for op_index, f in self._axis2feature[var_name].items():
                print("         visit op: {}".format(self._env.index2op[op_index]))
                for ff in f:
                    print("          feature: {}".format(ff))
        print("*************************************************")

    def print_cache(self):
        print("*************************************************")
        print("    ComputeVertex {} has axes:".format(self.op))
        for i, var_name in enumerate(self._cache_index2axis):
            print("    axis {}:".format(var_name))
            print("    state: {}".format(self._cache_axis2state[var_name]))
            for op_index, f in self._cache_axis2feature[var_name].items():
                print("         visit op: {}".format(self._env.index2op[op_index]))
                for ff in f:
                    print("          feature: {}".format(ff))
        print("*************************************************")

    def update_real_axes(self, old_var_name, new_axes):
        index = self._cache_axis2index[old_var_name]
        length = len(new_axes)
        new_var_names = [old_var_name + "_" + str(i) for i in range(length)]
        del self._cache_axis2index[old_var_name]
        del self._cache_index2axis[index]
        del self._cache_axis2real[old_var_name]
        for i, new_var_name in enumerate(new_var_names):
            self._cache_index2axis.insert(index + i, new_var_name)
            self._cache_axis2real[new_var_name] = new_axes[i]
        for i, var_name in enumerate(self._cache_index2axis):
            self._cache_axis2index[var_name] = i
        return new_var_names

    def update_axis_all(self, old_var_name, new_axes, new_features=None, new_states=None, new_doms=None):
        """
        During scheduling, features of axes may change, use this delete an old feature
        and add some new features.
        [NOTE]: All changes happen only to caches

        :param old_var_name: str
            axis_name to delete
        :param new_axes: list of IterVar
            use to bind new names to IterVars
        :param new_features: list of torch.Tensor
            use to bind new names to features, if not given, only update IterVars
        :return: new var names
        """
        index = self._cache_axis2index[old_var_name]
        length = len(new_axes)
        new_var_names = [old_var_name + "_" + str(i) for i in range(length)]
        del self._cache_axis2index[old_var_name]
        del self._cache_index2axis[index]
        del self._cache_axis2real[old_var_name]
        for i, new_var_name in enumerate(new_var_names):
            self._cache_index2axis.insert(index + i, new_var_name)
            self._cache_axis2real[new_var_name] = new_axes[i]
        for i, var_name in enumerate(self._cache_index2axis):
            self._cache_axis2index[var_name] = i
        if new_features:
            del self._cache_axis2feature[old_var_name]
            for i, new_var_name in enumerate(new_var_names):
                self._cache_axis2feature[new_var_name] = new_features[i]
        else:
            for i, new_var_name in enumerate(new_var_names):
                self._cache_axis2feature[new_var_name] = copy.deepcopy(self._cache_axis2feature[old_var_name])
            del self._cache_axis2feature[old_var_name]
        if new_states:
            del self._cache_axis2state[old_var_name]
            for i, new_var_name in enumerate(new_var_names):
                self._cache_axis2state[new_var_name] = new_states[i]
        else:
            for i, new_var_name in enumerate(new_var_names):
                self._cache_axis2state[new_var_name] = copy.deepcopy(self._cache_axis2state[old_var_name])
            del self._cache_axis2state[old_var_name]
        if new_doms:
            del self._cache_axis2dom[old_var_name]
            for i, new_var_name in enumerate(new_var_names):
                self._cache_axis2dom[new_var_name] = new_doms[i]
        else:
            total = len(new_axes)
            for i, new_var_name in enumerate(new_var_names):
                self._cache_axis2dom[new_var_name] = self._cache_axis2dom[old_var_name] // total
            del self._cache_axis2dom[old_var_name]
        return new_var_names

    def update_axes_order(self, new_order):
        """
        Change the order of axis

        :param new_order: list of str
            new axis order
        :return:  None
        """
        pos = []
        for axis in new_order:
            pos.append(self._cache_axis2index[axis])
        new_pos = sorted(pos)
        for i, p in enumerate(new_pos):
            self._cache_index2axis[p] = new_order[i]
            self._cache_axis2index[new_order[i]] = p


class VisitMessage(object):
    def __init__(self, op, var_names, axis2dom):
        self.op = op
        self.axis2dom = axis2dom
        self.features = {}
        for var_name in var_names:
            self.features[var_name] = {}    # from int to list of AxisFeature
        self.cur_var = None
        self.cur_exp = None
        self.vars = []
        self._Backup = namedtuple("Backup", ("last_exp", "last_var", "last_vars"))

        class AB(object):
            def __init__(self, var_name, a, b):
                self.var_name = var_name
                self.a = a
                self.b = b
        self._AB = AB
        self._backups = []

    def clear_(self):
        self.cur_var = None
        self.cur_exp = None
        self.vars = []

    def backup_(self):
        tmp = self._Backup(self.cur_exp, self.cur_var, self.vars)
        self._backups.append(tmp)

    def recover_(self):
        if self._backups:
            tmp = self._backups.pop()
            self.cur_exp = tmp.last_exp
            self.cur_var = tmp.last_var
            self.vars = tmp.last_vars

    def vars_append_(self, var_name):
        ab = self._AB(var_name, 1, 0)
        self.vars.append(ab)

    def get_dom(self, var_name):
        return self.axis2dom[var_name]


def visit_var(env, var, msg=None):
    msg.cur_var = var.name
    msg.vars_append_(var.name)


def visit_const_expr(env, const_expr, msg=None):
    if msg.cur_exp == "add":
        for ab in msg.vars:
            ab.b = const_expr.value
    if msg.cur_exp == "sub":
        for ab in msg.vars:
            ab.b = -const_expr.value
    elif msg.cur_exp == "mul":
        for ab in msg.vars:
            ab.a = const_expr.value
    elif msg.cur_exp == "div":
        for ab in msg.vars:
            ab.a = 1.0 / const_expr.value


def visit_add(env, add, msg=None):
    old_expr = msg.cur_exp
    msg.cur_exp = "add"
    visit_expr(env, add.a, msg)
    visit_expr(env, add.b, msg)
    msg.cur_exp = old_expr


def visit_sub(env, sub, msg=None):
    old_expr = msg.cur_exp
    msg.cur_exp = "sub"
    visit_expr(env, sub.a, msg)
    visit_expr(env, sub.b, msg)
    msg.cur_exp = old_expr


def visit_mul(env, mul, msg=None):
    old_expr = msg.cur_exp
    msg.cur_exp = "mul"
    visit_expr(env, mul.a, msg)
    visit_expr(env, mul.b, msg)
    msg.cur_exp = old_expr


def visit_div(env, div, msg=None):
    old_expr = msg.cur_exp
    msg.cur_exp = "div"
    visit_expr(env, div.a, msg)
    visit_expr(env, div.b, msg)
    msg.cur_exp = old_expr


def visit_and(env, _and, msg=None):
    old_expr = msg.cur_exp
    msg.cur_exp = "and"
    visit_expr(env, _and.a, msg)
    visit_expr(env, _and.b, msg)
    msg.cur_exp = old_expr


def visit_or(env, _or, msg=None):
    old_expr = msg.cur_exp
    msg.cur_exp = "or"
    visit_expr(env, _or.a, msg)
    visit_expr(env, _or.b, msg)
    msg.cur_exp = old_expr


def visit_not(env, _not, msg=None):
    old_expr = msg.cur_exp
    msg.cur_exp = "not"
    visit_expr(env, _not.a, msg)
    msg.cur_exp = old_expr


def visit_cmp_expr(env, cmp_expr, msg=None):
    old_expr = msg.cur_exp
    msg.cur_exp = "cmp"
    visit_expr(env, cmp_expr.a, msg)
    visit_expr(env, cmp_expr.b, msg)
    msg.cur_exp = old_expr


def visit_reduce(env, reduce, msg=None):
    for expr in reduce.source:
        visit_expr(env, expr, msg)


def visit_cast(env, cast, msg=None):
    visit_expr(env, cast.value, msg)


def visit_select(env, select, msg=None):
    visit_expr(env, select.true_value, msg)
    visit_expr(env, select.false_value, msg)


# Visit call is special because we care about it
def visit_call(env, call, msg=None):
    msg.backup_()
    msg.cur_exp = "call"
    used_op = call.func
    for i, expr in enumerate(call.args):
        msg.vars = []
        msg.cur_var = None
        expr = tvm.ir_pass.Simplify(expr)
        visit_expr(env, expr, msg)
        for ab in msg.vars:
            var_name = ab.var_name
            feature = AxisFeature(i, msg.get_dom(var_name), ab.a, ab.b)
            if env.op2index[used_op] not in msg.features[var_name]:
                msg.features[var_name][env.op2index[used_op]] = []
            msg.features[var_name][env.op2index[used_op]].append(feature)
    msg.recover_()


# These may be unimportant
def visit_let(env, let, msg=None):
    pass


def visit_ramp(env, ramp, msg=None):
    pass


def visit_load(env, load, msg=None):
    pass


def visit_shuffle(env, shuffle, msg=None):
    pass


def visit_broadcast(env, broadcast, msg=None):
    pass


# Visit Expr dispatches different visits
def visit_expr(env, expr, msg=None):
    p = tvm.expr
    next_steps = {
        p.Var: visit_var,
        p.IntImm: visit_const_expr,
        p.UIntImm: visit_const_expr,
        p.FloatImm: visit_const_expr,
        p.StringImm: visit_const_expr,
        p.Add: visit_add,
        p.Sub: visit_sub,
        p.Mul: visit_mul,
        p.Div: visit_div,
        p.CmpExpr: visit_cmp_expr,
        p.And: visit_and,
        p.Or: visit_or,
        p.Not: visit_not,
        p.Reduce: visit_reduce,
        p.Cast: visit_cast,
        p.Select: visit_select,
        p.Call: visit_call,
        p.Let: visit_let,
        p.Ramp: visit_ramp,
        p.Load: visit_load,
        p.Shuffle: visit_shuffle,
        p.Broadcast: visit_broadcast
    }
    next_step = next_steps[type(expr)]
    next_step(env, expr, msg)


# Visit Features starts here
def visit_feature(env, op, var_names, axis2dom):
    # currently only support computeOp
    if not isinstance(op, tvm.tensor.ComputeOp):
        return {}

    msg = VisitMessage(op, var_names, axis2dom)
    for body in op.body:
        msg.clear_()
        visit_expr(env, body, msg)
    return msg
