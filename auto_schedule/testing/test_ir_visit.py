import tvm
from collections import deque
from auto_schedule.utils import split_part_names


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


def visit(node, *args):
    visit_table = {
        tvm.stmt.LetStmt: visit_let_stmt,
        tvm.stmt.AssertStmt: visit_assert_stmt,
        tvm.stmt.ProducerConsumer: visit_producer_consumer,
        tvm.stmt.For: visit_for,
        tvm.stmt.Store: visit_store,
        tvm.stmt.Provide: visit_provide,
        tvm.stmt.Allocate: visit_allocate,
        tvm.stmt.AttrStmt: visit_attr_stmt,
        tvm.stmt.Free: visit_free,
        tvm.stmt.Realize: visit_realize,
        tvm.stmt.Block: visit_block,
        tvm.stmt.IfThenElse: visit_if_then_else,
        tvm.stmt.Evaluate: visit_evaluate,
        tvm.stmt.Prefetch: visit_prefetch,
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
        tvm.expr.Select: visit_select,
        tvm.expr.Load: visit_load,
        tvm.expr.Ramp: visit_ramp,
        tvm.expr.Broadcast: visit_broadcast,
        tvm.expr.Shuffle: visit_shuffle,
        tvm.expr.Call: visit_call,
        tvm.expr.Let: visit_let
    }
    which_type = type(node)
    func = visit_table[which_type]
    func(node, args[0])


def visit_let_stmt(node, ret):
    ret.append(node)
    visit(node.var, ret)
    visit(node.value, ret)
    visit(node.body, ret)


def visit_assert_stmt(node, ret):
    ret.append(node)
    visit(node.condition, ret)
    visit(node.message, ret)
    visit(node.body, ret)


def visit_producer_consumer(node, ret):
    ret.append(node)
    print(type(node), " func=", node.func)
    print(type(node), " is_producer=", node.is_producer)
    visit(node.body, ret)


def visit_for(node, ret):
    ret.append(node)
    visit(node.loop_var, ret)
    visit(node.min, ret)
    visit(node.extent, ret)
    print(type(node), " for_type=", node.for_type)
    print(type(node), " device_api=", node.device_api)
    visit(node.body, ret)


def visit_store(node, ret):
    ret.append(node)
    visit(node.buffer_var, ret)
    visit(node.value, ret)
    visit(node.index, ret)
    visit(node.predicate, ret)


def visit_provide(node, ret):
    ret.append(node)
    print(type(node), " func=", node.func)
    print(type(node), " value_index=", node.value_index)
    visit(node.value, ret)
    for arg in node.args:
        visit(arg, ret)


def visit_allocate(node, ret):
    ret.append(node)
    visit(node.buffer_var, ret)
    print(type(node), " dtype=", node.dtype)
    for extent in node.extents:
        visit(extent, ret)
    visit(node.condition, ret)
    visit(node.body, ret)


def visit_attr_stmt(node, ret):
    ret.append(node)
    print(type(node), " node=", node.node)
    print(type(node), " attr_key=", node.attr_key)
    visit(node.value, ret)
    visit(node.body, ret)


def visit_free(node, ret):
    ret.append(node)
    visit(node.buffer_var, ret)


def visit_realize(node, ret):
    ret.append(node)
    print(type(node), " func=", node.func)
    print(type(node), " value_index=", node.value_index)
    print(type(node), " dtype=", node.dtype)
    for bound in node.bounds:
        print(type(node), " bound=", bound)
    visit(node.condition, ret)
    visit(node.body, ret)


def visit_block(node, ret):
    ret.append(node)
    visit(node.first, ret)
    visit(node.rest, ret)


def visit_if_then_else(node, ret):
    ret.append(node)
    visit(node.condition, ret)
    visit(node.then_case, ret)
    visit(node.else_case, ret)


def visit_evaluate(node, ret):
    ret.append(node)
    visit(node.value, ret)


def visit_prefetch(node, ret):
    ret.append(node)
    print(type(node), " func=", node.func)
    print(type(node), " value_index=", node.value_index)
    print(type(node), " dtype=", node.dtype)
    for bound in node.bounds:
        print(type(node), " bound=", bound)


def visit_var(node, ret):
    ret.append(node)
    print(type(node), " name=", node.name)
    print(type(node), " dtype=", node.dtype)


def visit_reduce(node, ret):
    ret.append(node)
    print(type(node), " combiner=", node.combiner)
    for s in node.src:
        visit(s, ret)
    for r in node.rdom:
        print(type(node), " iter_var=", r)
    visit(node.condition, ret)
    print(type(node), " value_index=", node.value_index)


def visit_float_imm(node, ret):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    print(type(node), " value=", node.value)


def visit_int_imm(node, ret):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    print(type(node), " value=", node.value)


def visit_uint_imm(node, ret):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    print(type(node), " value=", node.value)


def visit_string_imm(node, ret):
    ret.append(node)
    print(type(node), " value=", node.value)


def visit_cast(node, ret):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    visit(node.value, ret)


def visit_add(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_sub(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_mul(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_div(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_mod(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_min(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_max(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_eq(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_ne(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_lt(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_le(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_gt(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_ge(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_and(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_or(node, ret):
    ret.append(node)
    visit(node.a, ret)
    visit(node.b, ret)


def visit_not(node, ret):
    ret.append(node)
    visit(node.a, ret)


def visit_select(node, ret):
    ret.append(node)
    visit(node.condition, ret)
    visit(node.true_value, ret)
    visit(node.false_value, ret)


def visit_load(node, ret):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    visit(node.buffer_var, ret)
    visit(node.index, ret)
    visit(node.predicate, ret)


def visit_ramp(node, ret):
    ret.append(node)
    visit(node.base, ret)
    print(type(node), " stride=", node.stride)
    print(type(node), " lanes=", node.lanes)


def visit_broadcast(node, ret):
    ret.append(node)
    visit(node.value, ret)
    print(type(node), " lanes=", node.lanes)


def visit_shuffle(node, ret):
    ret.append(node)
    for vec in node.vectors:
        visit(vec, ret)
    for index in node.indices:
        visit(index, ret)


def visit_call(node, ret):
    ret.append(node)
    print(type(node), " dtype=", node.dtype)
    print(type(node), " name=", node.name)
    for arg in node.args:
        visit(arg, ret)
    print(type(node), " call_type=", node.call_type)
    print(type(node), " func=", node.func)
    print(type(node), " value_index=", node.value_index)


def visit_let(node, ret):
    ret.append(node)
    visit(node.var, ret)
    visit(node.value, ret)
    visit(node.body, ret)


if __name__ == "__main__":
    from auto_schedule.examples import FUNC_TABLE
    func = FUNC_TABLE["conv3d_channel_batch"].func
    args = FUNC_TABLE["conv3d_channel_batch"].args
    op, bufs = func(*args)
    s = tvm.create_schedule(op)
    stmt = tvm.lower(s, bufs, simple_mode=True)
    print(stmt)
    ret = []
    visit(stmt, ret)
