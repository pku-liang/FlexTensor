import tvm


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


def visit_let_stmt(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " var=")
    visit(node.var, level+1)
    print("|  " * level + "|**" + " value=")
    visit(node.value, level+1)
    print("|  " * level + "|**" + " body=")
    visit(node.body, level+1)


def visit_assert_stmt(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " condition=")
    visit(node.condition, level+1)
    print("|  " * level + "|**" + " message=")
    visit(node.message, level+1)
    print("|  " * level + "|**" + " body=")
    visit(node.body, level+1)


def visit_producer_consumer(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " func=", node.func)
    print("|  " * level + "|**" + " is_producer=", node.is_producer)
    print("|  " * level + "|**" + " body=")
    visit(node.body, level+1)


def visit_for(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " loop_var=")
    visit(node.loop_var, level+1)
    print("|  " * level + "|**" + " min=")
    visit(node.min, level+1)
    print("|  " * level + "|**" + " extent=")
    visit(node.extent, level+1)
    print("|  " * level + "|**" + " for_type=", node.for_type)
    print("|  " * level + "|**" + " device_api=", node.device_api)
    print("|  " * level + "|**" + " body=")
    visit(node.body, level+1)


def visit_store(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " buffer_var=")
    visit(node.buffer_var, level+1)
    print("|  " * level + "|**" + " value=")
    visit(node.value, level+1)
    print("|  " * level + "|**" + " index=")
    visit(node.index, level+1)
    print("|  " * level + "|**" + " predicate=")
    visit(node.predicate, level+1)


def visit_provide(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " func=", node.func)
    print("|  " * level + "|**" + " value_index=", node.value_index)
    print("|  " * level + "|**" + " value=")
    visit(node.value, level+1)
    print("|  " * level + "|**" + " args=")
    for arg in node.args:
        visit(arg, level+1)


def visit_allocate(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " buffer_var=")
    visit(node.buffer_var, level+1)
    print("|  " * level + "|**" + " dtype=", node.dtype)
    print("|  " * level + "|**" + " extents=")
    for extent in node.extents:
        visit(extent, level+1)
    print("|  " * level + "|**" + " condition=")
    visit(node.condition, level+1)
    print("|  " * level + "|**" + " body=")
    visit(node.body, level+1)


def visit_attr_stmt(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " node=", type(node.node))
    print("|  " * level + "|**" + " attr_key=", node.attr_key)
    print("|  " * level + "|**" + " value=")
    visit(node.value, level+1)
    print("|  " * level + "|**" + " body=")
    visit(node.body, level+1)


def visit_free(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " buffer_var=")
    visit(node.buffer_var, level+1)


def visit_realize(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " func=", node.func)
    print("|  " * level + "|**" + " value_index=", node.value_index)
    print("|  " * level + "|**" + " dtype=", node.dtype)
    for bound in node.bounds:
        print("|  " * level + "|**" + " bound=", bound)
    print("|  " * level + "|**" + " condition=")
    visit(node.condition, level+1)
    print("|  " * level + "|**" + " body=")
    visit(node.body, level+1)


def visit_block(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " first=")
    visit(node.first, level+1)
    print("|  " * level + "|**" + " rest=")
    visit(node.rest, level+1)


def visit_if_then_else(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " condition=")
    visit(node.condition, level+1)
    print("|  " * level + "|**" + " then_case=")
    visit(node.then_case, level+1)
    if node.else_case:
        print("|  " * level + "|**" + " else_case=")
        visit(node.else_case, level+1)


def visit_evaluate(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " value=")
    visit(node.value, level+1)


def visit_prefetch(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " func=", node.func)
    print("|  " * level + "|**" + " value_index=", node.value_index)
    print("|  " * level + "|**" + " dtype=", node.dtype)
    for bound in node.bounds:
        print("|  " * level + "|**" + " bound=", bound)


def visit_var(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " name=", node.name)
    print("|  " * level + "|**" + " dtype=", node.dtype)


def visit_reduce(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " combiner=", node.combiner)
    print("|  " * level + "|**" + " src=")
    for s in node.src:
        visit(s, level+1)
    for r in node.rdom:
        print("|  " * level + "|**" + " iter_var=", r)
    print("|  " * level + "|**" + " condition=")
    visit(node.condition, level+1)
    print("|  " * level + "|**" + " value_index=", node.value_index)


def visit_float_imm(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " dtype=", node.dtype)
    print("|  " * level + "|**" + " value=", node.value)


def visit_int_imm(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " dtype=", node.dtype)
    print("|  " * level + "|**" + " value=", node.value)


def visit_uint_imm(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " dtype=", node.dtype)
    print("|  " * level + "|**" + " value=", node.value)


def visit_string_imm(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " value=", node.value)


def visit_cast(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " dtype=", node.dtype)
    print("|  " * level + "|**" + " value=")
    visit(node.value, level+1)


def visit_add(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level+1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level+1)


def visit_sub(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_mul(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_div(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_mod(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_min(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_max(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_eq(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_ne(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_lt(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_le(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_gt(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_ge(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_and(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_or(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)
    print("|  " * level + "|**" + " b=")
    visit(node.b, level + 1)


def visit_not(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " a=")
    visit(node.a, level + 1)


def visit_select(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " condition=")
    visit(node.condition, level+1)
    print("|  " * level + "|**" + " true_value=")
    visit(node.true_value, level+1)
    print("|  " * level + "|**" + " false_value=")
    visit(node.false_value, level+1)


def visit_load(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " dtype=", node.dtype)
    print("|  " * level + "|**" + " buffer_var=")
    visit(node.buffer_var, level+1)
    print("|  " * level + "|**" + " index=")
    visit(node.index, level+1)
    print("|  " * level + "|**" + " predicate=")
    visit(node.predicate, level+1)


def visit_ramp(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " base=")
    visit(node.base, level+1)
    print("|  " * level + "|**" + " stride=", node.stride)
    print("|  " * level + "|**" + " lanes=", node.lanes)


def visit_broadcast(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " value=")
    visit(node.value, level+1)
    print("|  " * level + "|**" + " lanes=", node.lanes)


def visit_shuffle(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " vectors=")
    for vec in node.vectors:
        visit(vec, level+1)
    print("|  " * level + "|**" + " indices=")
    for index in node.indices:
        visit(index, level+1)


def visit_call(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " dtype=", node.dtype)
    print("|  " * level + "|**" + " name=", node.name)
    print("|  " * level + "|**" + " args=")
    for arg in node.args:
        visit(arg, level+1)
    print("|  " * level + "|**" + " call_type=", node.call_type)
    print("|  " * level + "|**" + " func=", node.func)
    print("|  " * level + "|**" + " value_index=", node.value_index)


def visit_let(node, level):
    print("|  " * level + "|--", node.__class__)
    print("|  " * level + "|**" + " var=")
    visit(node.var, level+1)
    print("|  " * level + "|**" + " value=")
    visit(node.value, level+1)
    print("|  " * level + "|**" + " body=")
    visit(node.body, level+1)


if __name__ == "__main__":
    from auto_schedule.examples import FUNC_TABLE
    func = FUNC_TABLE["conv3d_channel_batch"].func
    args = FUNC_TABLE["conv3d_channel_batch"].args
    op, bufs = func(*args)
    s = tvm.create_schedule(op)
    stmt = tvm.lower(s, bufs, simple_mode=True)
    level = 0
    visit(stmt, level)
