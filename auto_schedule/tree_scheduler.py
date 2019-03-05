import tvm
import torch
from auto_schedule.utils import to_tuple


NODE_LIST = [tvm.stmt.LetStmt,
             tvm.stmt.AssertStmt,
             tvm.stmt.ProducerConsumer,
             tvm.stmt.For,
             tvm.stmt.Store,
             tvm.stmt.Provide,
             tvm.stmt.Allocate,
             tvm.stmt.AttrStmt,
             tvm.stmt.Free,
             tvm.stmt.Realize,
             tvm.stmt.Block,
             tvm.stmt.IfThenElse,
             tvm.stmt.Evaluate,
             tvm.stmt.Prefetch,
             tvm.expr.Var,
             tvm.expr.Reduce,
             tvm.expr.FloatImm,
             tvm.expr.IntImm,
             tvm.expr.UIntImm,
             tvm.expr.StringImm,
             tvm.expr.Cast,
             tvm.expr.Add,
             tvm.expr.Sub,
             tvm.expr.Mul,
             tvm.expr.Div,
             tvm.expr.Mod,
             tvm.expr.Min,
             tvm.expr.Max,
             tvm.expr.EQ,
             tvm.expr.NE,
             tvm.expr.LT,
             tvm.expr.LE,
             tvm.expr.GT,
             tvm.expr.GE,
             tvm.expr.And,
             tvm.expr.Or,
             tvm.expr.Not,
             tvm.expr.Select,
             tvm.expr.Load,
             tvm.expr.Ramp,
             tvm.expr.Broadcast,
             tvm.expr.Shuffle,
             tvm.expr.Call,
             tvm.expr.Let]


NODE_DICT = dict()
for i, node in enumerate(NODE_LIST):
    NODE_DICT[node] = i


def prepare_embedding(dim):
    length = len(NODE_LIST)
    embedding_layer = [torch.rand(dim, requires_grad=True) for i in range(length)]
    return embedding_layer


def visit(node, record, custom_func):
    visit_table = {
        tvm.stmt.LetStmt: visit_let_stmt,
        tvm.stmt.AssertStmt: visit_assert_stmt,
        tvm.stmt.ProducerConsumer: visit_producer_consumer,
        tvm.stmt.For: visit_for,
        tvm.stmt.Store: visit_store,
        tvm.stmt.Provide: visit_provide,
        tvm.stmt.Allocate: visit_allocate,
        tvm.stmt.AttrStmt: visit_attr_stmt,
        tvm.stmt.Block: visit_block,
        tvm.stmt.IfThenElse: visit_if_then_else,
        tvm.stmt.Evaluate: visit_evaluate,
        tvm.expr.Reduce: visit_reduce,
        tvm.expr.Cast: visit_cast,
        tvm.expr.Var: visit_var,
        tvm.expr.FloatImm: visit_const_expr,
        tvm.expr.IntImm: visit_const_expr,
        tvm.expr.UIntImm: visit_const_expr,
        tvm.expr.StringImm: visit_const_expr,
        tvm.expr.Add: visit_add,
        tvm.expr.Sub: visit_sub,
        tvm.expr.Mul: visit_mul,
        tvm.expr.Div: visit_div,
        tvm.expr.Mod: visit_mod,
        tvm.expr.Min: visit_min,
        tvm.expr.Max: visit_max,
        tvm.expr.EQ: visit_eq,
        tvm.expr.NE: visit_ne,
        tvm.expr.LT: visit_lt,
        tvm.expr.LE: visit_le,
        tvm.expr.GT: visit_gt,
        tvm.expr.GE: visit_ge,
        tvm.expr.And: visit_and,
        tvm.expr.Or: visit_or,
        tvm.expr.Not: visit_not,
        tvm.expr.Select: visit_select,
        tvm.expr.Load: visit_load,
        tvm.expr.Ramp: visit_ramp,
        tvm.expr.Broadcast: visit_broadcast,
        tvm.expr.Call: visit_call,
        tvm.expr.Let: visit_let
    }
    which_type = type(node)
    func = visit_table[which_type]
    return func(node, record, custom_func)


def visit_let_stmt(node, record, custom_func):
    a = visit(node.value, record, custom_func)
    b = visit(node.body, record, custom_func)
    feature = custom_func(node, [a, b])
    return feature


def visit_assert_stmt(node, record, custom_func):
    a = visit(node.condition, record, custom_func)
    b = visit(node.message, record, custom_func)
    c = visit(node.body, record, custom_func)
    feature = custom_func(node, [a, b, c])
    return feature


def visit_producer_consumer(node, record, custom_func):
    if node.is_producer:
        record.op_scope.append(node.func)
        record.add_op_msg(node.func)
    a = visit(node.body, record, custom_func)
    if node.is_producer:
        record.set_op_msg_item(node.func, "feature", a)
        record.op_scope.pop()
    feature = custom_func(node, [a])
    return feature


def visit_for(node, record, custom_func):
    record.add_iter_var(node.loop_var.name)
    record.set_iter_var_item(node.loop_var.name, "extent", node.extent.value)
    a = visit(node.body, record, custom_func)
    feature = custom_func(node, [a])
    return feature


def visit_store(node, record, custom_func):
    a = visit(node.value, record, custom_func)
    record.iter_var_buffer.clear()
    record.iter_var_factor = 1
    b = visit(node.index, record, custom_func)
    for iter_var_t in record.iter_var_buffer:
        record.add_iter_var_visit(iter_var_t[0], iter_var_t[1], iter_var_t[2])
    record.iter_var_buffer.clear()
    feature = custom_func(node, [a, b])
    return feature


def visit_provide(node, record, custom_func):
    tensor = node.func.output(node.value_index)
    shape = list(reversed(to_tuple(tensor.shape)))
    a = visit(node.value, record, custom_func)
    b = [a]
    cur = 1
    record.iter_var_buffer.clear()
    for i, arg in enumerate(reversed(node.args)):
        record.iter_var_factor = cur
        b.append(visit(arg, record, custom_func))
        cur *= shape[i]
    for iter_var_t in record.iter_var_buffer:
        record.add_iter_var_visit(iter_var_t[0], iter_var_t[1], iter_var_t[2])
    record.iter_var_buffer.clear()
    feature = custom_func(node, b)
    return feature


def visit_allocate(node, record, custom_func):
    record.set_buf_allocate(node.buffer_var.name, node.dtype, to_tuple(node.extents))
    a = visit(node.body, record, custom_func)
    feature = custom_func(node, [a])
    return feature


def visit_attr_stmt(node, record, custom_func):
    if node.attr_key == "storage_scope":
        record.add_buf_msg(node.node.name, node.value.value)
    a = visit(node.body, record, custom_func)
    feature = custom_func(node, [a])
    return feature


def visit_block(node, record, custom_func):
    a = visit(node.first, record, custom_func)
    b = visit(node.rest, record, custom_func)
    feature = custom_func(node, [a, b])
    return feature


def visit_if_then_else(node, record, custom_func):
    a = visit(node.condition, record, custom_func)
    b = visit(node.then_case, record, custom_func)
    c = visit(node.else_case, record, custom_func)
    feature = custom_func(node, [a, b, c])
    return feature


def visit_evaluate(node, record, custom_func):
    a = visit(node.value, record, custom_func)
    return custom_func(node, [a])


def visit_reduce(node, record, custom_func):
    a = []
    for s in node.src:
        a.append(visit(s, record, custom_func))
    a.append(visit(node.condition, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_cast(node, record, custom_func):
    a = visit(node.value, record, custom_func)
    feature = custom_func(node, [a])
    return feature


def visit_var(node, record, custom_func):
    if record.iter_var_factor:
        record.iter_var_buffer.append([node.name, 1, record.iter_var_factor])
    feature = custom_func(node, [])
    return feature


def visit_const_expr(node, record, custom_func):
    feature = custom_func(node, [])
    return feature


def visit_add(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_sub(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_mul(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
        if isinstance(operand, tvm.expr.ConstExpr) and record.iter_var_buffer:
                record.iter_var_buffer[-1][2] *= operand.value
    feature = custom_func(node, a)
    return feature


def visit_div(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
        if isinstance(operand, tvm.expr.ConstExpr) and record.iter_var_buffer:
            record.iter_var_buffer[-1][1] *= operand.value
    feature = custom_func(node, a)
    return feature


def visit_mod(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_min(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_max(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_eq(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_ne(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_lt(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_le(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_gt(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_ge(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_and(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_or(node, record, custom_func):
    operands = [node.a, node.b]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_not(node, record, custom_func):
    operands = [node.a]
    a = []
    for operand in operands:
        a.append(visit(operand, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_select(node, record, custom_func):
    a = visit(node.condition, record, custom_func)
    b = visit(node.true_value, record, custom_func)
    c = visit(node.false_value, record, custom_func)
    feature = custom_func(node, [a, b, c])
    return feature


def visit_load(node, record, custom_func):
    record.iter_var_buffer.clear()
    record.iter_var_factor = 1
    a = visit(node.index, record, custom_func)
    for iter_var_t in record.iter_var_buffer:
        record.add_iter_var_visit(iter_var_t[0], iter_var_t[1], iter_var_t[2])
    record.iter_var_buffer.clear()
    b = visit(node.predicate, record, custom_func)
    feature = custom_func(node, [a, b])
    return feature


def visit_ramp(node, record, custom_func):
    a = visit(node.base, record, custom_func)
    feature = custom_func(node, [a])
    return feature


def visit_broadcast(node, record, custom_func):
    a = visit(node.value, record, custom_func)
    feature = custom_func(node, [a])
    return feature


def visit_call(node, record, custom_func):
    a = []
    for arg in node.args:
        a.append(visit(arg, record, custom_func))
    feature = custom_func(node, a)
    return feature


def visit_let(node, record, custom_func):
    a = visit(node.value, record, custom_func)
    b = visit(node.body, record, custom_func)
    feature = custom_func(node, [a, b])
    return feature


class OpMessage(object):
    def __init__(self):
        self.feature = None
        self.iter_var_name_msg_dict = dict()
        self.iter_var_name_lst = []
        self.cur_iter_var_name_lst = []
        self.shape = []

    def __str__(self):
        ret = "OpMessage(\n"
        ret += "    feature=" + str(self.feature) + "\n"
        ret += "    iter_var message:" + "\n"
        for key, value in self.iter_var_name_msg_dict.items():
            ret += "      name=" + key + "\n    message=\n    " + str(value) + "\n"
        ret += "    iter_var name list: " + str(self.iter_var_name_lst) + "\n"
        ret += "    shape=" + str(self.shape) + "\n"
        ret += ")\n"
        return ret

    def set_iter_var_item(self, iter_var_name, name, value):
        setattr(self.iter_var_name_msg_dict[iter_var_name], name, value)

    def add_iter_var(self, iter_var_name):
        pos = len(self.iter_var_name_lst)
        self.iter_var_name_lst.append(iter_var_name)
        self.iter_var_name_msg_dict[iter_var_name] = IterVarMessage()
        self.set_iter_var_item(iter_var_name, "pos", pos)

    def add_iter_var_visit(self, iter_var_name, pre, factor):
        self.iter_var_name_msg_dict[iter_var_name].visit.append((pre, factor))


class BufMessage(object):
    def __init__(self, storage_scope):
        self.dtype = None
        self.storage_scope = storage_scope
        self.ranges = None

    def __str__(self):
        ret = "BufMessage(dtype=" + self.dtype + ", storage_scope=" + self.storage_scope + ", ranges=" + str(
            self.ranges) + ")"
        return ret


class IterVarMessage(object):
    def __init__(self):
        self.extent = None
        self.pos = None
        self.visit = []
        self.split_factor_lst = []

    def __str__(self):
        ret = "IterVarMessage(extent={},pos={},visit={})".format(self.extent, self.pos, self.visit)
        return ret


class Record(object):
    def __init__(self):
        self.tree_feature = None
        self.buf_msg_dict = dict()
        self.op_msg_dict = dict()
        self.name_op_dict = dict()
        self.op_scope = []
        self.iter_var_buffer = []
        self.iter_var_factor = None
        self.cur_op_lst = []

    def print(self):
        print("tree_feature:", self.tree_feature)
        print("buffer message:")
        for key, value in self.buf_msg_dict.items():
            print("    buffer_name=", key)
            print("    message=", str(value))
        print("operation message:")
        for key, value in self.op_msg_dict.items():
            print("    op=", key)
            print("    message=\n    ", str(value))
        print("operation name:")
        for key, value in self.name_op_dict.items():
            print("    name=", key)
            print("    op=", value)

    def set_tree_feature(self, tree_feature):
        self.tree_feature = tree_feature

    def set_buf_allocate(self, buf_var_name, dtype, ranges):
        self.buf_msg_dict[buf_var_name].dtype = dtype
        self.buf_msg_dict[buf_var_name].ranges = ranges

    def add_buf_msg(self, buf_var_name, scope):
        self.buf_msg_dict[buf_var_name] = BufMessage(scope)

    def set_op_msg_item(self, op, name, value):
        setattr(self.op_msg_dict[op], name, value)

    def set_iter_var_item(self, iter_var_name, name, value):
        self.op_msg_dict[self.op_scope[-1]].set_iter_var_item(iter_var_name, name, value)

    def add_iter_var_visit(self, iter_var_name, pre, factor):
        self.op_msg_dict[self.op_scope[-1]].add_iter_var_visit(iter_var_name, pre, factor)

    def add_op_msg(self, op):
        self.name_op_dict[op.name] = op
        self.op_msg_dict[op] = OpMessage()
        self.op_msg_dict[op].shape = to_tuple(op.output(0).shape)

    def add_iter_var(self, iter_var_name):
        self.op_msg_dict[self.op_scope[-1]].add_iter_var(iter_var_name)


class OpState(object):
    def __init__(self):
        self.schedule = []
        self.outer_block = []
        self.inner_block = []
        self.middle_block = []
        self.compute_inline = False
        self.iter_var_dict = dict()
        self.shared_cache = []
        self.local_cache = []
        self.spatial_iter_var_name_lst = []
        self.spatial_done_lst = []
        self.reduce_iter_var_name_lst = []
        self.reduce_done_lst = []
        
    def get_iter_var(self, name):
        return self.iter_var_dict[name]

    def prepare_iter_var(self, op):
        for iter_var in op.axis:
            self.iter_var_dict[iter_var.var.name] = iter_var
            self.spatial_iter_var_name_lst.append(iter_var.var.name)
        for iter_var in op.reduce_axis:
            self.iter_var_dict[iter_var.var.name] = iter_var
            self.reduce_iter_var_name_lst.append(iter_var.var.name)

    def replace(self, iter_var, iter_var_lst):
        del self.iter_var_dict[iter_var.var.name]
        for iter_var in iter_var_lst:
            self.iter_var_dict[iter_var.var.name] = iter_var


class Action(object):
    def __init__(self, type_key, a, b):
        self.type_key = type_key
        self.a = a
        self.b = b

    def to_json(self):
        return {
            "type_key": self.type_key,
            "a": self.a,
            "b": self.b
        }


class Target(object):
    def __init__(self, type_key, feature):
        self.type = type_key
        self.feature = feature
        self.support_compute_inline = True
        self.support_split = True
        self.support_reorder = True
        self.support_compute_at = True
        self.support_cache_read = False
        self.support_cache_write = False
        self.support_parallel = False
        self.support_unroll = True
        self.support_vectorize = True
        self.support_bind = False
        self.support_rfactor = False
        self.support_double_buffer = True

    def turn_on(self, name):
        setattr(self, name, True)

    def turn_down(self, name):
        setattr(self, name, False)

    def to_json(self):
        return {
            "type": self.type,
            "feature": self.feature
        }


class ComputeState(object):
    def __init__(self, compute_name, args, states):
        self.compute_name = compute_name
        self.args = args
        self.op_action_dict = dict()
        for op_name, state in states.items():
            self.op_action_dict[op_name] = state.schedule

    def to_json(self):
        op_action_dict = dict()
        for op_name, action_lst in self.op_action_dict.items():
            lst = []
            for a in action_lst:
                lst.append(a.to_json())
            op_action_dict[op_name] = lst
        return {
            "compute_name": self.compute_name,
            "args": self.args,
            "op_action_dict": op_action_dict
        }


class Sample(object):
    def __init__(self, compute_state, target, op, type_key, choice, reward):
        self.compute_state = compute_state
        self.target = target
        self.op = op
        self.type_key = type_key
        self.choice = choice
        self.reward = reward

    def to_json(self):
        return {
            "compute_state": self.compute_state.to_json(),
            "target": self.target.to_json(),
            "op": self.op.name,
            "type_key": self.type_key,
            "choice": self.choice,
            "reward": self.reward
        }


def restore_compute(s, bfs_order, op_action_dict):
    op_dict = dict()
    states = dict()
    for op in bfs_order:
        if isinstance(op, tvm.tensor.ComputeOp):
            op_dict[op.name] = op
            states[op.name] = OpState()
            states[op.name].prepare_iter_var(op)
            action_lst = op_action_dict[op.name]
            for action in action_lst:
                if action.type == "compute_inline":
                    s[op].compute_inline()
                elif action.type == "split":
                    iter_var_name = action.a
                    factor = action.b
                    iter_var = states[op.name].get_iter_var(iter_var_name)
                    outer, inner = s[op].split(iter_var, factor=factor)
                    states[op.name].replace(iter_var, [outer, inner])
                elif action.type == "reorder":
                    iter_var_name_lst = action.a
                    iter_var_lst = [states[op.name].get_iter_var(x) for x in iter_var_name_lst]
                    s[op].reorder(*iter_var_lst)
    return s








