import tvm
import torch
import math
from auto_schedule.utils import to_tuple, permute, interleave, gen_group, nearest_power_of_two
from auto_schedule.models import OpScheduleCPUd5, OpScheduleGPUd5
from collections import deque


BASE = 1000
FINF = float("inf")
DIM = 4


class VarMessage(object):
    def __init__(self, name, factor):
        self.name = name
        self.reverse = False
        self.under_mod = False
        self.factor = factor
        self.div = 1
        self.mod = float("inf")
        self.visit_op = None


class Record(object):
    def __init__(self):
        self.in_call = False
        self.visit_result = dict()


def visit(expr, record):
    visit_table = {
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
    which_type = type(expr)
    func = visit_table[which_type]
    return func(expr, record)


def visit_reduce(expr, record):
    lst1, lst2 = [], []
    for s in expr.source:
        a_lst, av = visit(s, record)
        lst1 += a_lst
        lst2 += av
    return lst1, lst2


def visit_cast(expr, record):
    return visit(expr.value, record)


def visit_var(expr, record):
    if record.in_call:
        return [VarMessage(expr.name, 1)], []
    else:
        return [], []


def visit_const_expr(expr, record):
    if record.in_call:
        return [], [expr.value]
    else:
        return [], []


def visit_add(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_sub(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    for it in b_lst:
        it.factor *= -1
    for i in range(len(bv)):
        bv[i] *= -1
    return a_lst + b_lst, av + bv


# TODO multiplication is handled without consideration for case such as "i*j"
def visit_mul(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    tmp_lst = []
    for ait in a_lst:
        for bit in b_lst:
            tmp_lst.append(VarMessage(ait.name, ait.factor * bit.factor))
            tmp_lst.append(VarMessage(bit.name, bit.factor * ait.factor))
    for ait in a_lst:
        for v in bv:
            tmp_lst.append(VarMessage(ait.name, ait.factor * v))
    for bit in b_lst:
        for v in av:
            tmp_lst.append(VarMessage(bit.name, bit.factor * v))
    tmp_v = []
    for v1 in av:
        for v2 in bv:
            tmp_v.append(v1 * v2)
    return tmp_lst, tmp_v


# TODO it's hard to determine how to handle division case
def visit_div(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    tmp_lst = []
    for ait in a_lst:
        for bit in b_lst:
            vm = VarMessage(ait.name, ait.factor)
            vm.div = bit.factor
            tmp_lst.append(vm)
            bvm = VarMessage(bit.name, vm.factor)
            bvm.div = vm.div
            bvm.reverse = True
            tmp_lst.append(bvm)
    for ait in a_lst:
        for v in bv:
            vm = VarMessage(ait.name, ait.factor)
            vm.div = v
            tmp_lst.append(vm)
    for bit in b_lst:
        for v in av:
            vm = VarMessage(bit.name, v)
            vm.div = bit.factor
            vm.reverse = True
            tmp_lst.append(vm)
    tmp_v = []
    for v1 in av:
        for v2 in bv:
            tmp_v.append(v1 // v2)
    return tmp_lst, tmp_v


# TODO it's hard to determine how to handle modular case
def visit_mod(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    tmp_lst = []
    for ait in a_lst:
        for bit in b_lst:
            vm = VarMessage(ait.name, ait.factor)
            vm.mod = bit.factor
            tmp_lst.append(vm)
    for ait in a_lst:
        for v in bv:
            vm = VarMessage(ait.name, ait.factor)
            vm.mod = v
            tmp_lst.append(vm)
    for bit in b_lst:
        vm = VarMessage(bit.name, bit.factor)
        vm.under_mod = True
        tmp_lst.append(vm)
    tmp_v = []
    for v1 in av:
        for v2 in bv:
            tmp_v.append(v1 % v2)
    return tmp_lst, tmp_v


def visit_min(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_max(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_eq(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_ne(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_lt(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_le(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_gt(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_ge(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_and(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_or(expr, record):
    a_lst, av = visit(expr.a, record)
    b_lst, bv = visit(expr.b, record)
    return a_lst + b_lst, av + bv


def visit_not(expr, record):
    a_lst, av = visit(expr.a, record)
    return a_lst, av


def visit_select(expr, record):
    a_lst, av = visit(expr.condition, record)
    b_lst, bv = visit(expr.true_value, record)
    c_lst, cv = visit(expr.false_value, record)
    return a_lst + b_lst + c_lst, av + bv + cv


def visit_load(expr, record):
    a_lst, av = visit(expr.index, record)
    b_lst, bv = visit(expr.predicate, record)
    return a_lst + b_lst, av + bv


def visit_ramp(expr, record):
    return visit(expr.base, record)


def visit_broadcast(expr, record):
    return visit(expr.value, record)


def visit_call(expr, record):
    if not expr.func in record.visit_result:
        record.visit_result[expr.func] = []
        for arg in expr.args:
            record.visit_result[expr.func].append(([], []))
    record.in_call = True
    for i, arg in enumerate(expr.args):
        lst, vlst = visit(arg, record)
        if lst and not vlst:
            vlst.append(0)
        record.visit_result[expr.func][i][0].extend(lst)
        record.visit_result[expr.func][i][1].extend(vlst)
    record.in_call = False
    return [], []


def visit_let(expr, record):
    a_lst, av = visit(expr.value, record)
    b_lst, bv = visit(expr.body, record)
    return a_lst + b_lst, av + bv


def graph_analysis(ops):
    bfs_order = []
    down_graph = {}
    visited = set()
    q = deque()
    for op in ops:
        q.append(op)
        visited.add(op)
    while q:
        cur = q.popleft()
        bfs_order.append(cur)
        for t in cur.input_tensors:
            if t.op not in visited:
                visited.add(t.op)
                q.append(t.op)
            if t.op not in down_graph:
                down_graph[t.op] = []
            down_graph[t].append(cur)
    return bfs_order, down_graph


def get_axis_feature(op_lst):
    op_result_dict = dict()

    class OpResult(object):
        def __init__(self):
            self.iter_var_feature = dict()
            self.visit_feature = []

    for op in op_lst:
        op_result_dict[op] = OpResult()

    for op in op_lst:
        iter_var_dom_dict = dict()
        if hasattr(op, 'axis'):
            for iter_var in op.axis:
                iter_var_dom_dict[iter_var.var.name] = iter_var.dom.extent.value
            for iter_var in op.reduce_axis:
                iter_var_dom_dict[iter_var.var.name] = iter_var.dom.extent.value
        iter_var_feature = dict()
        if hasattr(op, 'body'):
            for bi in op.body:
                record = Record()
                visit(bi, record)
                for visit_op, res in record.visit_result.items():
                    visit_lst = []
                    factors = []
                    cur = 1
                    for v in reversed(to_tuple(visit_op.output(0).shape)):
                        factors.append(cur)
                        cur *= v
                    factors = list(reversed(factors))
                    for i, (lst, vlst) in enumerate(res):
                        tmp_dict = dict()
                        for vm in lst:
                            if vm.name not in tmp_dict:
                                tmp_dict[vm.name] = []
                            tmp_dict[vm.name].append(
                                [iter_var_dom_dict[vm.name], factors[i], vm.factor, vm.div,
                                 vm.mod if vm.mod != float("inf") else -1, int(vm.reverse),
                                 int(vm.under_mod)])
                        for name, value in tmp_dict.items():
                            if name not in iter_var_feature:
                                iter_var_feature[name] = []
                            iter_var_feature[name].extend(value)
                        visit_lst.append((tmp_dict, vlst))
                    op_result_dict[visit_op].visit_feature.append((op, visit_lst))
        op_result_dict[op].iter_var_feature = iter_var_feature
    return op_result_dict


def op_schedule_cpu_general_dx(dim, s, op, model, random=False, sampling=True):
    ret_dict = dict()
    record = Record()
    if len(op.body) != 1:
        raise ValueError("only support one output operation")
    visit(op.body[0], record)
    # iter_var_name -> extent
    iter_var_dom_dict = dict()
    if hasattr(op, 'axis'):
        for iter_var in op.axis:
            iter_var_dom_dict[iter_var.var.name] = iter_var.dom.extent.value
        for iter_var in op.reduce_axis:
            iter_var_dom_dict[iter_var.var.name] = iter_var.dom.extent.value
    # iter_var_name -> feature: (op_name, lst)
    visit_trace_dict = dict()
    iter_var_feature = dict()
    iter_var_feature["none"] = []
    for visit_op, res in record.visit_result.items():
        # prepare factors from shape
        factors = []
        cur = 1
        for v in reversed(to_tuple(visit_op.output(0).shape)):
            factors.append(cur)
            cur *= v
        factors = list(reversed(factors))
        # every dimension
        for i, (lst, vlst) in enumerate(res):
            tmp_dict = dict()
            for vm in lst:
                if vm.name not in tmp_dict:
                    tmp_dict[vm.name] = []
                if vm.name not in visit_trace_dict:
                    visit_trace_dict[vm.name] = []
                tmp_dict[vm.name].append((visit_op.name, [iter_var_dom_dict[vm.name], factors[i], vm.factor]))
                visit_trace_dict[vm.name].append((visit_op.name, i))
            for name, value in tmp_dict.items():
                if name not in iter_var_feature:
                    iter_var_feature[name] = []
                iter_var_feature[name].extend(value)
    # shape of inputs
    input_shapes = dict()
    for t in op.input_tensors:
        tmp = list(to_tuple(t.shape))
        cha = dim - len(tmp)
        if cha < 0:
            raise ValueError("Dimension of Operation should <= {}".format(dim))
        input_shapes[t.op.name] = [1] * cha + [x for x in tmp]
    # align
    # spatial
    cha = dim - len(op.axis)
    if cha < 0:
        raise ValueError("Dimension of Operation should <= {}".format(dim))
    spatial_axis_names = ["none"] * cha
    spatial_iter_vars = [None] * cha
    spatial_axis_extents = [1] * cha
    spatial_iter_var_dict = dict()
    for i, iter_var in enumerate(op.axis):
        spatial_axis_names.append(iter_var.var.name)
        spatial_iter_vars.append(iter_var)
        spatial_axis_extents.append(iter_var.dom.extent.value)
        spatial_iter_var_dict[iter_var.var.name] = i + cha
    # reduce
    cha = dim - len(op.reduce_axis)
    if cha < 0:
        raise ValueError("Reduce dimension should <= {}".format(dim))
    reduce_axis_names = ["none"] * cha
    reduce_iter_vars = [None] * cha
    reduce_axis_extents = [1] * cha
    reduce_iter_var_dict = dict()
    for i, iter_var in enumerate(op.reduce_axis):
        reduce_axis_names.append(iter_var.var.name)
        reduce_iter_vars.append(iter_var)
        reduce_axis_extents.append(iter_var.dom.extent.value)
        reduce_iter_var_dict[iter_var.var.name] = i + cha
    # spatial split
    split_candidates = spatial_axis_names
    split_decision, value_lst = model("spatial", split_candidates, spatial_axis_extents, iter_var_feature, input_shapes, random, sampling)
    ret_dict["spatial"] = value_lst
    spatial_split_factors = [f for f in split_decision]
    spatial_outer_extents = [math.ceil(spatial_axis_extents[i] / spatial_split_factors[i]) for i in range(dim)]
    # reduce
    reduce_candidates = reduce_axis_names
    split_decision, value_lst = model("reduce", reduce_candidates, reduce_axis_extents, iter_var_feature, input_shapes, random, sampling)
    ret_dict["reduce"] = value_lst
    reduce_split_factors = [f for f in split_decision]
    reduce_outer_extents = [math.ceil(reduce_axis_extents[i] / reduce_split_factors[i]) for i in range(dim)]
    # reorder
    # part one
    part_one_iter_var_names = ["none" if spatial_outer_extents[i] == 1 else spatial_axis_names[i] for i in range(dim)]
    part_one_extents = dict()
    for i in range(dim):
        part_one_extents[part_one_iter_var_names[i]] = spatial_outer_extents[i]
    reorder_candidates = permute(part_one_iter_var_names)
    reorder_choice, logits = model("reorder_one", part_one_iter_var_names, part_one_extents, iter_var_feature, input_shapes, random, sampling)
    ret_dict["reorder_one"] = logits
    reorder_decision = reorder_candidates[reorder_choice]
    part_one_order = reorder_decision
    # parallel
    parallel_candidates = gen_group(reorder_decision, padding="none")
    parallel_extents = part_one_extents
    parallel_choice, logits = model("parallel", reorder_decision, parallel_extents, iter_var_feature, input_shapes, random, sampling)
    parallel_decision = parallel_candidates[parallel_choice]
    ret_dict["parallel"] = logits
    use_parallel = [True if dec != "none" else False for dec in parallel_decision]
    # part two
    part_two_iter_var_names = ["none" if reduce_outer_extents[i] == 1 else reduce_axis_names[i] for i in range(dim)]
    part_two_extents = dict()
    for i in range(dim):
        part_two_extents[part_two_iter_var_names[i]] = reduce_outer_extents[i]
    reorder_candidates = permute(part_two_iter_var_names)
    reorder_choice, logits = model("reorder_two", part_two_iter_var_names, part_two_extents, iter_var_feature, input_shapes, random, sampling)
    reorder_decision = reorder_candidates[reorder_choice]
    ret_dict["reorder_two"] = logits
    part_two_order = reorder_decision
    # part three
    part_three_names_a = ["none" if spatial_split_factors[i] == 1 else spatial_axis_names[i] for i in range(dim)]
    part_three_names_b = ["none" if reduce_split_factors[i] == 1 else reduce_axis_names[i] for i in range(dim)]
    reorder_candidates = interleave(part_three_names_a, part_three_names_b)
    part_three_extents = dict()
    for i in range(dim):
        part_three_extents[part_three_names_a[i]] = spatial_split_factors[i]
        part_three_extents[part_three_names_b[i]] = reduce_split_factors[i]
    reorder_choice, logits = model("reorder_three", part_three_names_a + part_three_names_b, part_three_extents, iter_var_feature, input_shapes, random, sampling)
    reorder_decision = reorder_candidates[reorder_choice]
    ret_dict["reorder_three"] = logits
    part_three_order = reorder_decision

    # record the diary
    diary = dict()
    diary["spatial"] = spatial_outer_extents
    diary["spatial_factors"] = spatial_split_factors
    diary["reorder_one"] = part_one_order
    diary["parallel"] = parallel_decision
    diary["reduce"] = reduce_outer_extents
    diary["reduce_factors"] = reduce_split_factors
    diary["reorder_two"] = part_two_order
    diary["reorder_three"] = part_three_order

    # real schedule
    # cache write
    LocalCache = s.cache_write(op.output(0), "local")
    cha = dim - len(op.axis)
    if cha < 0:
        raise ValueError("Dimension of Operation should <= {}".format(dim))
    cache_spatial_iter_vars = [None] * cha
    for i, iter_var in enumerate(s[LocalCache].op.axis):
        cache_spatial_iter_vars.append(iter_var)
    cha = dim - len(op.reduce_axis)
    if cha < 0:
        raise ValueError("Reduce dimension should <= {}".format(dim))
    cache_reduce_iter_vars = [None] * cha
    for i, iter_var in enumerate(s[LocalCache].op.reduce_axis):
        cache_reduce_iter_vars.append(iter_var)
    # spatial split
    spatial_outer, spatial_inner = [None] * dim, [None] * dim
    for i in range(dim):
        if spatial_iter_vars[i] is not None:
            if spatial_split_factors[i] != 1 and spatial_split_factors[i] != spatial_axis_extents[i]:
                outer, inner = s[op].split(spatial_iter_vars[i], factor=spatial_split_factors[i])
                spatial_outer[i] = outer
                spatial_inner[i] = inner
            elif spatial_split_factors[i] == 1:
                spatial_outer[i] = spatial_iter_vars[i]
            else:
                spatial_inner[i] = spatial_iter_vars[i]
    order = []
    for iter_var in spatial_outer + spatial_inner:
        if iter_var is not None:
            order.append(iter_var)
    if order:
        s[op].reorder(*order)
    # spatial reorder
    real_reorder = []
    last_index = -1
    last_iter_var = None
    for name in part_one_order:
        if name != "none":
            index = spatial_iter_var_dict[name]
            iter_var = spatial_outer[index]
            if iter_var is None:
                raise ValueError("unexpected NoneType")
            last_index = index
            last_iter_var = iter_var
            real_reorder.append(iter_var)
    if real_reorder:
        s[op].reorder(*real_reorder)
    # fuse
    fuse_lst = []
    for i in range(dim):
        name = parallel_decision[i]
        if name != "none":
            if use_parallel[i]:
                index = spatial_iter_var_dict[name]
                iter_var = spatial_outer[index]
                if iter_var is None:
                    raise ValueError("unexpected NoneType")
                if index == last_index:
                    last_index = -2
                fuse_lst.append(iter_var)
    if len(fuse_lst) > 1:
        parallel_iter_var = s[op].fuse(*fuse_lst)
    elif len(fuse_lst) == 1:
        parallel_iter_var = fuse_lst[0]
    else:
        parallel_iter_var = None
    # parallel
    if parallel_iter_var is not None:
        s[op].parallel(parallel_iter_var)
    # compute_at
    if last_index == -2:
        last_iter_var = parallel_iter_var
    if last_iter_var is not None:
        s[LocalCache].compute_at(s[op], last_iter_var)
    # reduce split
    reduce_outer = [None] * dim
    reduce_inner = [None] * dim
    for i in range(dim):
        if reduce_axis_names[i] != "none":
            if reduce_split_factors[i] != 1 and reduce_outer_extents[i] != 1:
                outer, inner = s[LocalCache].split(cache_reduce_iter_vars[i], factor=reduce_split_factors[i])
                reduce_outer[i] = outer
                reduce_inner[i] = inner
            elif reduce_split_factors[i] == 1:
                reduce_outer[i] = cache_reduce_iter_vars[i]
            else:
                reduce_inner[i] = cache_reduce_iter_vars[i]
    # reduce reorder
    real_order = []
    is_reduce = []
    for name in part_two_order:
        if name != "none":
            index = reduce_iter_var_dict[name]
            iter_var = reduce_outer[index]
            if iter_var is None:
                raise ValueError("unexpected NoneType")
            real_order.append(iter_var)
            is_reduce.append(True)
    # inner reorder
    for name in part_three_order:
        if name != "none":
            if name in spatial_iter_var_dict:
                index = spatial_iter_var_dict[name]
                iter_var = cache_spatial_iter_vars[index]
                if iter_var is None:
                    raise ValueError("unexpected NoneType")
                real_order.append(iter_var)
                is_reduce.append(False)
            elif name in reduce_iter_var_dict:
                index = reduce_iter_var_dict[name]
                iter_var = reduce_inner[index]
                if iter_var is None:
                    raise ValueError("unexpected NoneType")
                real_order.append(iter_var)
                is_reduce.append(True)
    if real_order:
        s[LocalCache].reorder(*real_order)
    # unroll and  vectorize
    p = len(real_order) - 1
    while p >= 0:
        if not is_reduce[p]:
            s[LocalCache].vectorize(real_order[p])
            p -= 1
            break
        p -= 1
    if p >= 0:
        s[LocalCache].unroll(real_order[p])
    return ret_dict, diary


def op_schedule_gpu_general_dx(dim, s, op, model, random=False, sampling=True):
    ret_dict = dict()
    record = Record()
    if len(op.body) != 1:
        raise ValueError("only support one output operation")
    visit(op.body[0], record)
    # iter_var_name -> extent
    iter_var_dom_dict = dict()
    if hasattr(op, 'axis'):
        for iter_var in op.axis:
            iter_var_dom_dict[iter_var.var.name] = iter_var.dom.extent.value
        for iter_var in op.reduce_axis:
            iter_var_dom_dict[iter_var.var.name] = iter_var.dom.extent.value
    # iter_var_name -> feature: (op_name, lst)
    visit_trace_dict = dict()
    iter_var_feature = dict()
    iter_var_feature["none"] = []
    for visit_op, res in record.visit_result.items():
        # prepare factors from shape
        factors = []
        cur = 1
        for v in reversed(to_tuple(visit_op.output(0).shape)):
            factors.append(cur)
            cur *= v
        factors = list(reversed(factors))
        # every dimension
        for i, (lst, vlst) in enumerate(res):
            tmp_dict = dict()
            for vm in lst:
                if vm.name not in tmp_dict:
                    tmp_dict[vm.name] = []
                if vm.name not in visit_trace_dict:
                    visit_trace_dict[vm.name] = []
                tmp_dict[vm.name].append((visit_op.name, [iter_var_dom_dict[vm.name], factors[i], vm.factor]))
                visit_trace_dict[vm.name].append((visit_op.name, i))
            for name, value in tmp_dict.items():
                if name not in iter_var_feature:
                    iter_var_feature[name] = []
                iter_var_feature[name].extend(value)
    # shape of inputs
    input_shapes = dict()
    for t in op.input_tensors:
        tmp = list(to_tuple(t.shape))
        cha = dim - len(tmp)
        if cha < 0:
            raise ValueError("Dimension of Operation should <= {}".format(dim))
        input_shapes[t.op.name] = [1] * cha + [x for x in tmp]
    # align
    # spatial
    cha = dim - len(op.axis)
    if cha < 0:
        raise ValueError("Dimension of Operation should <= {}".format(dim))
    spatial_axis_names = ["none"] * cha
    spatial_iter_vars = [None] * cha
    spatial_axis_extents = [1] * cha
    spatial_iter_var_dict = dict()
    for i, iter_var in enumerate(op.axis):
        spatial_axis_names.append(iter_var.var.name)
        spatial_iter_vars.append(iter_var)
        spatial_axis_extents.append(iter_var.dom.extent.value)
        spatial_iter_var_dict[iter_var.var.name] = i + cha
    # reduce
    cha = dim - len(op.reduce_axis)
    if cha < 0:
        raise ValueError("Reduce dimension should <= {}".format(dim))
    reduce_axis_names = ["none"] * cha
    reduce_iter_vars = [None] * cha
    reduce_axis_extents = [1] * cha
    reduce_iter_var_dict = dict()
    for i, iter_var in enumerate(op.reduce_axis):
        reduce_axis_names.append(iter_var.var.name)
        reduce_iter_vars.append(iter_var)
        reduce_axis_extents.append(iter_var.dom.extent.value)
        reduce_iter_var_dict[iter_var.var.name] = i + cha
    # spatial split
    # part one
    split_candidates = spatial_axis_names
    spatial_split_decision, value_lst = model("spatial_one", split_candidates, spatial_axis_extents, iter_var_feature, input_shapes, random, sampling)
    ret_dict["spatial_one"] = value_lst
    spatial_outer_extents = [spatial_axis_extents[i] // spatial_split_decision[i] for i in range(dim)]
    # part two
    spatial_middle_extents = [2 if spatial_split_decision[i] % 2 == 0 else 1 for i in range(dim)]
    spatial_left_extents = [spatial_axis_extents[i] // (spatial_outer_extents[i] * spatial_middle_extents[i]) for i in range(dim)]
    # part three
    split_candidates = spatial_axis_names
    spatial_split_decision, value_lst = model("spatial_three", split_candidates, spatial_left_extents, iter_var_feature, input_shapes, random, sampling)
    ret_dict["spatial_three"] = value_lst
    spatial_inner_extents = [spatial_left_extents[i] // spatial_split_decision[i] for i in range(dim)]
    # left extents
    spatial_left_extents = [spatial_split_decision[i] for i in range(dim)]
    # reduce
    reduce_candidates = reduce_axis_names
    reduce_split_decision, value_lst = model("reduce", reduce_candidates, reduce_axis_extents, iter_var_feature, input_shapes, random, sampling)
    ret_dict["reduce"] = value_lst
    # outer extents
    reduce_outer_extents = [reduce_axis_extents[i] // reduce_split_decision[i] for i in range(dim)]
    # left extents
    reduce_left_extents = [reduce_split_decision[i] for i in range(dim)]
    # reorder
    # part one
    part_one_iter_var_names = spatial_axis_names
    part_one_extents = dict()
    for i in range(dim):
        part_one_extents[part_one_iter_var_names[i]] = spatial_outer_extents[i]
    reorder_candidates = permute(part_one_iter_var_names)
    reorder_choice, logits = model("reorder_one", part_one_iter_var_names, part_one_extents, iter_var_feature, input_shapes, random, sampling)
    ret_dict["reorder_one"] = logits
    reorder_decision = reorder_candidates[reorder_choice]
    part_one_order = reorder_decision
    # part two
    part_two_iter_var_names = reduce_axis_names
    part_two_extents = dict()
    for i in range(dim):
        part_two_extents[part_two_iter_var_names[i]] = reduce_outer_extents[i]
    reorder_candidates = permute(part_two_iter_var_names)
    reorder_choice, logits = model("reorder_two", part_two_iter_var_names, part_two_extents, iter_var_feature, input_shapes, random, sampling)
    reorder_decision = reorder_candidates[reorder_choice]
    ret_dict["reorder_two"] = logits
    part_two_order = reorder_decision
    # part three
    part_three_names_a = spatial_axis_names
    part_three_names_b = reduce_axis_names
    reorder_candidates = interleave(part_three_names_a, part_three_names_b)
    part_three_extents = dict()
    for i in range(dim):
        part_three_extents[part_three_names_a[i]] = spatial_left_extents[i]
        part_three_extents[part_three_names_b[i]] = reduce_left_extents[i]
    reorder_choice, logits = model("reorder_three", part_three_names_a + part_three_names_b, part_three_extents, iter_var_feature, input_shapes, random, sampling)
    reorder_decision = reorder_candidates[reorder_choice]
    ret_dict["reorder_three"] = logits
    part_three_order = reorder_decision

    # hack here
    # spatial_outer_extents = [1, 1, 128, 1, 1]
    # spatial_middle_extents = [1, 1, 1, 1, 1]
    # spatial_inner_extents = [1, 1, 4, 7, 7]
    # spatial_left_extents = [1, 1, 2, 1, 1]
    # part_one_order = ["none", "b", "k", "i", "j"]
    # reduce_outer_extents = [1, 1, 128, 1, 1]
    # reduce_left_extents = [1, 1, 8, 3, 3]
    # part_two_order = ["none", "none", "rc", "ry", "rx"]
    # part_three_order = ["none", "none", "rc", "ry", "rx", "none", "b", "k", "i", "j"]
    # record the diary
    part_one_order = spatial_axis_names
    part_two_order = reduce_axis_names
    part_three_order = reduce_axis_names + spatial_axis_names
    diary = dict()
    diary["spatial_one"] = spatial_outer_extents
    diary["spatial_two"] = spatial_middle_extents
    diary["spatial_three"] = spatial_inner_extents
    diary["spatial_four"] = spatial_left_extents
    diary["reorder_one"] = part_one_order
    diary["reduce"] = reduce_outer_extents
    diary["reduce_inner"] = reduce_left_extents
    diary["reorder_two"] = part_two_order
    diary["reorder_three"] = part_three_order

    # real schedule
    # cache read
    input_pos_dict = dict()
    read_cache_share_lst = []
    for i, t in enumerate(op.input_tensors):
        input_pos_dict[t.op.name] = i
        cache = s.cache_read(t, "shared", [op])
        read_cache_share_lst.append(cache)
    read_cache_local_lst = []
    for t in read_cache_share_lst:
        cache = s.cache_read(t, "local", [op])
        read_cache_local_lst.append(cache)
    read_cache_spatial_iter_vars_lst = []
    for cache in read_cache_share_lst:
        cha = dim - len(s[cache].op.axis)
        if cha < 0:
            raise ValueError("Dimension of Operation should <= {}".format(dim))
        read_cache_spatial_iter_vars = [None] * cha
        for i, iter_var in enumerate(s[cache].op.axis):
            read_cache_spatial_iter_vars.append(iter_var)
        read_cache_spatial_iter_vars_lst.append(read_cache_spatial_iter_vars)

    # cache write
    LocalCache = s.cache_write(op.output(0), "local")
    cha = dim - len(op.axis)
    if cha < 0:
        raise ValueError("Dimension of Operation should <= {}".format(dim))
    cache_spatial_iter_vars = [None] * cha
    for i, iter_var in enumerate(s[LocalCache].op.axis):
        cache_spatial_iter_vars.append(iter_var)
    cha = dim - len(op.reduce_axis)
    if cha < 0:
        raise ValueError("Reduce dimension should <= {}".format(dim))
    cache_reduce_iter_vars = [None] * cha
    for i, iter_var in enumerate(s[LocalCache].op.reduce_axis):
        cache_reduce_iter_vars.append(iter_var)
    # spatial split
    # part one
    spatial_outer, spatial_left = [None] * dim, [None] * dim
    for i in range(dim):
        if spatial_iter_vars[i] is not None:
            # # print("check split 1", i, spatial_iter_vars[i], spatial_outer_extents[i])
            outer, inner = s[op].split(spatial_iter_vars[i], nparts=spatial_outer_extents[i])
            spatial_outer[i] = outer
            spatial_left[i] = inner
    # part two
    spatial_middle, spatial_remain = [None] * dim, [None] * dim
    for i in range(dim):
        if spatial_left[i] is not None:
            # # print("check split 2", i, spatial_left[i], spatial_middle_extents[i])
            outer, inner = s[op].split(spatial_left[i], nparts=spatial_middle_extents[i])
            spatial_middle[i] = outer
            spatial_remain[i] = inner
    # part three
    spatial_inner, spatial_last = [None] * dim, [None] * dim
    for i in range(dim):
        if spatial_remain[i] is not None:
            # # print("check split 3", i, spatial_remain[i], spatial_inner_extents[i])
            outer, inner = s[op].split(spatial_remain[i], nparts=spatial_inner_extents[i])
            spatial_inner[i] = outer
            spatial_last[i] = inner
    order = []
    for iter_var in spatial_outer + spatial_middle + spatial_inner + spatial_last:
        if iter_var is not None:
            order.append(iter_var)
    if order:
        # # print("check order", order)
        s[op].reorder(*order)
    # spatial reorder
    outer_reorder = []
    middle_reorder = []
    inner_reorder = []
    for name in part_one_order:
        if name != "none":
            index = spatial_iter_var_dict[name]
            iter_var = spatial_outer[index]
            if iter_var is None:
                raise ValueError("unexpected NoneType")
            outer_reorder.append(iter_var)
            # the following iter_var may be NoneType
            iter_var = spatial_middle[index]
            if iter_var is not None:
                middle_reorder.append(iter_var)
            iter_var = spatial_inner[index]
            if iter_var is not None:
                inner_reorder.append(iter_var)
    if outer_reorder:
        # print("check order one", outer_reorder)
        s[op].reorder(*outer_reorder)
    if middle_reorder:
        # print("check order two", middle_reorder)
        s[op].reorder(*middle_reorder)
    if inner_reorder:
        # print("check order three", inner_reorder)
        s[op].reorder(*inner_reorder)
    # bind
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    vthread_x = tvm.thread_axis("vthread", name="vx")
    vthread_y = tvm.thread_axis("vthread", name="vy")
    vthread_z = tvm.thread_axis("vthread", name="vz")
    thread_x = tvm.thread_axis("threadIdx.x")
    thread_y = tvm.thread_axis("threadIdx.y")
    thread_z = tvm.thread_axis("threadIdx.z")
    block_extents = [1, 1, 1]
    thread_extents = [1, 1, 1]
    pos = dim - 1
    # blockIdx.x, vthreadx, threadIdx.x
    while pos >= 0:
        iter_var_name = part_one_order[pos]
        if iter_var_name != "none":
            index = spatial_iter_var_dict[iter_var_name]
            if spatial_outer[index] is not None:
                # print("check bind bx", pos, iter_var_name, spatial_outer[index])
                block_extents[2] = spatial_outer_extents[index]
                s[op].bind(spatial_outer[index], block_x)
                if spatial_middle[index] is not None:
                    # print("check bind vx", pos, iter_var_name, spatial_middle[index])
                    s[op].bind(spatial_middle[index], vthread_x)
                if spatial_inner[index] is not None:
                    # print("check bind tx", pos, iter_var_name, spatial_inner[index])
                    thread_extents[2] = spatial_inner_extents[index]
                    s[op].bind(spatial_inner[index], thread_x)
                pos -= 1
                break
        pos -= 1
    # blockIdx.y, vthready, threadIdx.y
    while pos >= 0:
        iter_var_name = part_one_order[pos]
        if iter_var_name != "none":
            index = spatial_iter_var_dict[iter_var_name]
            if spatial_outer[index] is not None:
                # print("check bind by", pos, iter_var_name, spatial_outer[index])
                block_extents[1] = spatial_outer_extents[pos]
                s[op].bind(spatial_outer[index], block_y)
                if spatial_middle[index] is not None:
                    # print("check bind vy", pos, iter_var_name, spatial_middle[index])
                    s[op].bind(spatial_middle[index], vthread_y)
                if spatial_inner[index] is not None:
                    # print("check bind ty", pos, iter_var_name, spatial_inner[index])
                    thread_extents[1] = spatial_inner_extents[index]
                    s[op].bind(spatial_inner[index], thread_y)
                pos -= 1
                break
        pos -= 1
    # blockIdx.z, fuse
    fuse_lst_bz = []
    fuse_lst_vz = []
    fuse_lst_tz = []
    while pos >= 0:
        iter_var_name = part_one_order[pos]
        if iter_var_name != "none":
            index = spatial_iter_var_dict[iter_var_name]
            if spatial_outer[index] is not None:
                fuse_lst_bz.append(spatial_outer[index])
                block_extents[0] *= spatial_outer_extents[index]
                if spatial_middle[index] is not None:
                    fuse_lst_vz.append(spatial_middle[index])
                if spatial_inner[index] is not None:
                    fuse_lst_tz.append(spatial_inner[index])
                    thread_extents[0] *= spatial_inner_extents[index]
        pos -= 1
    if fuse_lst_bz:
        # print("check fuse bz", fuse_lst_bz)
        fused = s[op].fuse(*list(reversed(fuse_lst_bz)))
        s[op].bind(fused, block_z)
    if fuse_lst_vz:
        # print("check fuse vz", fuse_lst_vz)
        fused = s[op].fuse(*list(reversed(fuse_lst_vz)))
        s[op].bind(fused, vthread_z)
    if fuse_lst_tz:
        # print("check fuse tz", fuse_lst_tz)
        fused = s[op].fuse(*list(reversed(fuse_lst_tz)))
        s[op].bind(fused, thread_z)

    # print("check extents")

    # local write cache compute_at
    cache_write_axis = None
    candidates = [spatial_outer, spatial_middle, spatial_inner]

    candidate_pos = 2
    while cache_write_axis is None and candidate_pos >= 0:
        pos = dim - 1
        while pos >= 0:
            name = part_one_order[pos]
            if name != "none":
                index = spatial_iter_var_dict[name]
                if candidates[candidate_pos][index] is not None:
                    # print("check compute at", name, candidate_pos, pos, index, candidates[candidate_pos][index])
                    cache_write_axis = candidates[candidate_pos][index]
                    break
            pos -= 1
        candidate_pos -= 1

    if cache_write_axis is None:
        raise RuntimeError("unexpected NoneType")
    s[LocalCache].compute_at(s[op], cache_write_axis)
    # reduce split
    reduce_outer = [None] * dim
    reduce_inner = [None] * dim
    for i in range(dim):
        if reduce_axis_names[i] != "none":
            outer, inner = s[LocalCache].split(cache_reduce_iter_vars[i], nparts=reduce_outer_extents[i])
            reduce_outer[i] = outer
            reduce_inner[i] = inner
    # reduce reorder
    real_order_part_two = []
    is_reduce_part_two = []
    for name in part_two_order:
        if name != "none":
            index = reduce_iter_var_dict[name]
            iter_var = reduce_outer[index]
            if iter_var is None:
                raise ValueError("unexpected NoneType")
            real_order_part_two.append(iter_var)
            is_reduce_part_two.append(True)
    # print("check reorder two", real_order_part_two)
    # inner reorder
    real_order_part_three = []
    is_reduce_part_three = []
    real_order_part_three_extents = []
    for name in part_three_order:
        if name != "none":
            if name in spatial_iter_var_dict:
                index = spatial_iter_var_dict[name]
                iter_var = cache_spatial_iter_vars[index]
                if iter_var is None:
                    raise ValueError("unexpected NoneType")
                real_order_part_three.append(iter_var)
                is_reduce_part_three.append(False)
            elif name in reduce_iter_var_dict:
                index = reduce_iter_var_dict[name]
                iter_var = reduce_inner[index]
                if iter_var is None:
                    raise ValueError("unexpected NoneType")
                real_order_part_three.append(iter_var)
                is_reduce_part_three.append(True)
            real_order_part_three_extents.append(part_three_extents[name])
    # print("check reorder three", real_order_part_three)
    real_order = real_order_part_two + real_order_part_three
    if real_order:
        s[LocalCache].reorder(*real_order)

    # read cache compute at
    for i in range(len(op.input_tensors)):
        scache = read_cache_share_lst[i]
        lcache = read_cache_local_lst[i]
        if len(real_order_part_two) < 1:
            s[scache].compute_at(s[LocalCache], inner_reorder[-1])
        else:
            s[scache].compute_at(s[LocalCache], real_order_part_two[-1])
        s[lcache].compute_at(s[LocalCache], real_order_part_three[0])
        for cache in [scache, lcache]:
            fuse_lst = []
            extent = 1
            for axis in s[cache].op.axis:
                fuse_lst.append(axis)
                extent *= axis.dom.extent.value
            # print("check extent", extent, fuse_lst)
            fused = s[cache].fuse(*fuse_lst)
            if extent > thread_extents[0]:
                # print("check cache split 1", thread_extents[0])
                outer, fused = s[cache].split(fused, nparts=thread_extents[0])
                s[cache].bind(outer, thread_z)
                extent = extent // thread_extents[0]
            if extent > thread_extents[1]:
                # print("check cache split 2", thread_extents[1])
                outer, fused = s[cache].split(fused, nparts=thread_extents[1])
                s[cache].bind(outer, thread_y)
                extent = extent // thread_extents[1]
            if extent > thread_extents[2]:
                # print("check cache split 3", thread_extents[2])
                outer, fused = s[cache].split(fused, nparts=thread_extents[2])
                s[cache].bind(outer, thread_x)
                extent = extent // thread_extents[2]
            # outer, inner = s[cache].split(fused, factor=min(extent, 4))
            # s[cache].vectorize(inner)

    # alive = []
    # alive_extents = []
    # for lst in read_cache_spatial_iter_vars_lst:
    #     tmp = [False] * len(lst)
    #     tmp_ext = [0] * len(lst)
    #     alive.append(tmp)
    #     alive_extents.append(tmp_ext)
    # for name in part_three_order[1:]:
    #     if name == "none":
    #         continue
    #     trace = visit_trace_dict[name]
    #     for (op_name, pos) in trace:
    #         index = input_pos_dict[op_name]
    #         lst = alive[index]
    #         lst[pos] = True
    #         lst = alive_extents[index]
    #         lst[pos] = max(lst[pos], part_three_extents[name])
    # if len(real_order_part_three) > 0:
    #     for i in range(len(op.input_tensors)):
    #         scache = read_cache_share_lst[i]
    #         lcache = read_cache_local_lst[i]
    #         s[scache].compute_at(s[LocalCache], real_order_part_three[0])
    #         if len(real_order_part_three) == 1:
    #             s[lcache].compute_inline()
    #         else:
    #             s[lcache].compute_at(s[LocalCache], real_order_part_three[1])
    #         alive_lst = alive[i]
    #         alive_extents_lst = alive_extents[i]
    #         point = dim - 1
    #         while point >= 0:
    #             if alive_lst[point]:
    #                 parts = thread_extents[1]
    #                 if parts == 0:
    #                     parts = 32
    #                 if alive_extents_lst[point] >= parts > 1:
    #                     outer, inner = s[scache].split(s[scache].op.axis[point], nparts=parts)
    #                     s[scache].bind(outer, thread_x)
    #                     _, inner = s[scache].split(inner, factor=min(nearest_power_of_two(alive_extents_lst[point] // parts), 4))
    #                     s[scache].vectorize(inner)
    #                 point -= 1
    #                 break
    #             point -= 1
    #         while point >= 0:
    #             if alive_lst[point]:
    #                 parts = thread_extents[0]
    #                 if parts == 0:
    #                     parts = 16
    #                 outer, inner = s[scache].split(s[scache].op.axis[point], nparts=parts)
    #                 s[scache].bind(outer, thread_y)
    #                 point -= 1
    #                 break
    #             point -= 1
    # else:
    #     for i in range(len(op.input_tensors)):
    #         scache = read_cache_share_lst[i]
    #         lcache = read_cache_local_lst[i]
    #         s[scache].compute_inline()
    #         s[lcache].compute_inline()

    # unroll and  vectorize
    if len(real_order_part_three) > 1:
        p = len(real_order_part_three) - 1
        # while p >= 2:
        #     if not is_reduce_part_three[p]:
        #         # print("check vec", p)
        #         _, inner = s[LocalCache].split(real_order_part_three[p], factor=min(nearest_power_of_two(real_order_part_three_extents[p]), 4))
        #         s[LocalCache].vectorize(inner)
        #         p -= 1
        #         break
        #     p -= 1
        if p >= 1:
            # print("check unroll", p)
            # _, inner = s[LocalCache].split(real_order_part_three[p], factor=min(real_order_part_three_extents[p], 128))
            s[LocalCache].unroll(real_order_part_three[p])
    return ret_dict, diary


def able_inline(op, down_graph):
    is_compute = isinstance(op, tvm.tensor.ComputeOp)
    has_reduce = hasattr(op, "reduce_axis") and op.reduce_axis
    is_output = False
    for i in range(op.num_outputs):
        if op.output(i) not in down_graph:
            is_output = True
            break
    return is_compute and (not has_reduce) and (not is_output)


def graph_schedule_cpu_general_dx(dim, s, ops, model_path, random=False, sampling=True):
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    bfs_order, down_graph = graph_analysis(ops)
    group_points = []
    for op in bfs_order:
        if not isinstance(op, tvm.tensor.ComputeOp):
            continue
        if able_inline(op, down_graph):
            s[op].compute_inline()
        else:
            group_points.append(op)
    model = OpScheduleCPUd5(3, 128)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    _, diary = op_schedule_cpu_general_dx(dim, s, group_points[0], model, random, sampling)
    for ele in diary:
        print(ele)
    # for op in group_points:
    #     op_schedule_cpu_general_dx(dim, s, op, model, device, random)


def graph_schedule_gpu_general_dx(dim, s, ops, model_path, random=False, sampling=True):
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    bfs_order, down_graph = graph_analysis(ops)
    group_points = []
    for op in bfs_order:
        if not isinstance(op, tvm.tensor.ComputeOp):
            continue
        if able_inline(op, down_graph):
            s[op].compute_inline()
        else:
            group_points.append(op)
    model = OpScheduleGPUd5(3, 128)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    _, diary = op_schedule_gpu_general_dx(dim, s, group_points[0], model, random, sampling)
    for name, ele in diary.items():
        print(name, ele)
    # for op in group_points:
    #     op_schedule_cpu_general_dx(dim, s, op, model, device, random)
