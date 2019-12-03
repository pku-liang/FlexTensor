import tvm
from collections import deque
from auto_schedule.utils import split_part_names


def visit(node, ):
    visit_table = {
        tvm.expr.Var: visit_var,
        tvm.expr.Reduce: visit_reduce,
        tvm.expr.FloatImm: visit_float_imm,
        tvm.expr.IntImm: visit_int_imm,
        tvm.expr.UIntImm: visit_uint_imm,
        tvm.expr.StringImm: visit_string_imm,
        tvm.expr.Cast: visit_cast,
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
        # tvm.expr.Select: visit_select,
        # tvm.expr.Load: visit_load,
        # tvm.expr.Ramp: visit_ramp,
        # tvm.expr.Broadcast: visit_broadcast,
        # tvm.expr.Shuffle: visit_shuffle,
        tvm.expr.Call: visit_call,
        # tvm.expr.Let: visit_let
    }
    which_type = type(node)
    func = visit_table[which_type]
    func(node, *args)


def visit_var(node, *args):
    ret.append(node)
    print(type(node), " name=", node.name)
    print(type(node), " dtype=", node.dtype)


def visit_reduce(node, *args):
    # print(type(node), " combiner=", node.combiner)
    for s in node.source:
        visit(s, *args)
    for r in node.axis:
        print(type(node), " iter_var=", r)
    visit(node.condition, *args)
    print(type(node), " value_index=", node.value_index)


def visit_float_imm(node, *args):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    print(type(node), " value=", node.value)


def visit_int_imm(node, *args):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    print(type(node), " value=", node.value)


def visit_uint_imm(node, *args):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    print(type(node), " value=", node.value)


def visit_string_imm(node, *args):
    ret.append(node)
    print(type(node), " value=", node.value)


def visit_cast(node, *args):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    visit(node.value, *args)


def visit_add(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_sub(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_mul(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_div(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_mod(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_min(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_max(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_eq(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_ne(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_lt(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_le(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_gt(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_ge(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_and(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_or(node, *args):
    ret.append(node)
    visit(node.a, *args)
    visit(node.b, *args)


def visit_not(node, *args):
    ret.append(node)
    visit(node.a, *args)


# def visit_select(node, *args):
#     ret.append(node)
#     visit(node.condition, *args)
#     visit(node.true_value, *args)
#     visit(node.false_value, *args)


# def visit_load(node, *args):
#     ret.append(node)
#     print(type(node), " dtype=", node.dtype)
#     visit(node.buffer_var, *args)
#     visit(node.index, *args)
#     visit(node.predicate, *args)


# def visit_ramp(node, *args):
#     ret.append(node)
#     visit(node.base, *args)
#     print(type(node), " stride=", node.stride)
#     print(type(node), " lanes=", node.lanes)


# def visit_broadcast(node, *args):
#     ret.append(node)
#     visit(node.value, *args)
#     print(type(node), " lanes=", node.lanes)


# def visit_shuffle(node, *args):
#     ret.append(node)
#     for vec in node.vectors:
#         visit(vec, *args)
#     for index in node.indices:
#         visit(index, *args)


def visit_call(node, *args):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    print(type(node), " name=", node.name)
    for arg in node.args:
        visit(arg, *args)
    print(type(node), " call_type=", node.call_type)
    print(type(node), " func=", node.func)
    print(type(node), " value_index=", node.value_index)


# def visit_let(node, *args):
#     ret.append(node)
#     visit(node.var, *args)
#     visit(node.value, *args)
#     visit(node.body, *args)


class ExprVisitor(object):
    def visit(self, expr):
        dispatcher = {
        tvm.expr.Var: visit_var,
        tvm.expr.Reduce: visit_reduce,
        tvm.expr.FloatImm: visit_float_imm,
        tvm.expr.IntImm: visit_int_imm,
        tvm.expr.UIntImm: visit_uint_imm,
        tvm.expr.StringImm: visit_string_imm,
        tvm.expr.Cast: visit_cast,
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
        tvm.expr.Call: visit_call}
        type_info = type(expr)
        if type_info == tvm.expr.Reduce:
            self.visit_reduce(expr)
        elif type_info == tvm.expr.Call:
            self.visit_call(expr)
        elif type_info == tvm.expr.Var:
            self.visit_var(expr)
        elif type_info == tvm.expr.FloatImm:
            self.visit_float_imm(expr)
        elif type_info == tvm.expr.IntImm:
            self.visit_int_imm(expr)
        elif type_info == tvm.expr.UIntImm:
            self.visit_uint_imm(expr)
        elif type_info == tvm.expr.StringImm:
            self.visit_string_imm(expr)
        elif type_info == tvm.expr.Add:
            self.visit_add(expr)
        elif type_info == tvm.expr.Sub:
            self.visit_sub(expr)
        elif type_info == tvm.expr.Mul:
            self.visit_mul(expr)
        elif type_info == tvm.expr.Div:
            self.visit_div(expr)
        elif type_info == tvm.expr.Mod:
            self.visit_mod(expr)
        elif type_info == tvm.expr.And:
            self.visit_and(expr)
        elif type_info == tvm.expr.Or:
            self.visit_or(expr)
        elif type_info == tvm.expr.NOt:
            self.visit_not(expr)
        elif type_info == tvm.expr.Cast:
            self.visit_cast(expr)
        elif type_info == tvm.expr.EQ:
            self.visit_eq(expr)
        elif type_info == tvm.expr.NE:
            self.visit_ne(expr)
        elif type_info == tvm.expr.LT:
            self.visit_lt(expr)
        elif type_info == tvm.expr.LE:
            self.visit_le(expr)
        elif type_info == tvm.expr.GT:
            self.visit_gt(expr)
        elif type_info == tvm.expr.GE:
            self.visit_ge(expr)
        elif type_info == tvm.expr.Min:
            self.visit_min(expr)
        elif type_info == tvm.expr.Max:
            self.visit_max(expr)
        

    def visit_reduce(self):
        pass

    def visit_call(self):
        pass

    def visit_var(self):
        pass 

    def visit_float_imm(self):
        pass

    def visit_int_imm(self):
        pass

    def visit_uint_imm(self):
        pass

    def visit_string_imm(self):
        pass

    def visit_add(self):
        pass

    def visit_sub(self):
        pass

    def visit_div(self):
        pass

    def visit_mul(self):
        pass

    def visit_mod(self):
        pass

    def visit_and(self):
        pass

    def visit_or(self):
        pass

    def visit_not(self):
        pass

    def visit_min(self):
        pass

    def visit_max(self):
        pass

    def visit_eq(self):
        pass

    def visit_ne(self):
        pass

    def visit_le(self):
        pass
    
    def visit_lt(self):
        pass

    def visit_ge(self):
        pass

    def visit_gt(self):
        pass

    def visit_cast(self):
        pass




if __name__ == "__main__":
    from auto_schedule.examples import FUNC_TABLE
    func = FUNC_TABLE["conv3d_channel_batch"].func
    args = FUNC_TABLE["conv3d_channel_batch"].args
    op, bufs = func(*args)
    print(op.reduce_axis)
    s = tvm.create_schedule(op)
    stmt = tvm.lower(s, bufs, simple_mode=True)
    # print(stmt)
    ret = []
    # visit(stmt, *args)
    print(op.input_tensors[0].op.body)
    print(op.input_tensors[1].op.shape[0].value)
    print(op.axis[0].var.name == op.body[0].source[0].a.args[0].name)
    print(op.body[0].source[0].a.call_type)
    print(type(op.input_tensors[0].op.body[0].call_type), dir(op.body[0]))
    a = tvm.placeholder([4, 4])
    b = tvm.compute([4, 4], lambda i, j: a[(i * 4 + j) * 3 - i * 12 - j * 3 + i, j])
    print(tvm.ir_pass.Simplify(b.op.body[0].args[0]))
    print(op.axis[0].var == op.body[0].source[0].a.args[0])
