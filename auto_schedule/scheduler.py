import tvm
import numpy as np
import torch
import math
import time
import multiprocessing as multi
from auto_schedule.utils import to_tuple, permute, interleave, gen_group
from auto_schedule.models import OpScheduleCPUd5, ScheduleModel
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


def evaluate(s, bufs, target, dev_id, number=10, timeout=10.0):
    proc = []
    q = multi.Queue()
    for i in range(number):
        p = multi.Process(target=_evaluate, args=(s, bufs, target, dev_id, 1, q))
        p.start()
        proc.append(p)
    beg = time.time()
    while time.time() - beg < timeout:
        if any(p.is_alive() for p in proc):
            time.sleep(.1)
        else:
            break
    else:
        for p in proc:
            p.terminate()
            p.join()
    count = 0
    sum = 0
    while not q.empty():
        sum += q.get()
        count += 1
    while count < number:
        sum += timeout * 1000.0
        count += 1
    return sum / count


def _evaluate(s, bufs, target, dev_id, number=1, q=None):
    func = tvm.build(s, bufs, target)
    ctx = tvm.context(target, dev_id)
    tvm_arys = []
    for arg in bufs:
        shape = to_tuple(arg.shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(arg.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
    time_cost = evaluator(*tvm_arys).mean * 1e3
    if q:
        q.put(time_cost)
    return time_cost


class Target(object):
    def __init__(self, type_key, bx=1, by=1, bz=1, tx=1, ty=1, tz=1, vtx=FINF, vty=FINF, vtz=FINF):
        self.type = type_key
        self.bx, self.by, self.bz = bx, by, bz
        self.tx, self.ty, self.tz = tx, ty, tz
        self.vtx, self.vty, self.vtz = vtx, vty, vtz
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


class Schedule(object):
    def __init__(self, ops, target):
        if not isinstance(ops, (list, tuple)):
            ops = [ops]
        self.ops = ops
        self.target = target
        self.block_factors = [target.bx, target.by, target.bz]
        self.thread_factors = [target.tx, target.ty, target.tz]
        self.vthread_factors = [target.vtx, target.vty, target.vtz]
        self.reduce_factors = []

        bfs_order, down_graph = graph_analysis(ops)
        self.bfs_order = bfs_order
        self.down_graph = down_graph
        self.need_bind = target.support_bind

        class OpSchedule(object):
            def __init__(self, op):
                self.op = op
                self.shape = to_tuple(op.output(0).shape)
                self.compute_inline = False
                self.is_compute = isinstance(op, tvm.tensor.ComputeOp)
                self.has_reduce = self.is_compute and bool(op.reduce_axis)
                self.able_inline = self.is_compute and not self.has_reduce      # can't inline reduction
                self.next_num = [0 for i in range(op.num_outputs)]
                self.is_output = True
                for i in range(op.num_outputs):
                    if i > 0:
                        self.able_inline = False       # can only inline compute op with 1 output
                    if op.output(i) not in down_graph:
                        self.able_inline = False       # should has consumer
                    else:
                        self.is_output = False
                        self.next_num[i] = len(down_graph[op.output(i)])
                self.num_outputs = op.num_outputs
                self.output_tensors = [op.output(i) for i in range(op.num_outputs)]
                if self.is_compute:
                    self.input_tensors = op.input_tensors
                else:
                    self.input_tensors = []
                self.num_inputs = len(self.input_tensors)
                self.need_cache_read = self.is_compute and self.num_inputs > 0
                self.need_cache_write = self.is_compute and self.num_outputs > 0 and self.is_output
                self.use_write_share_cache = [False for i in range(self.num_outputs)]
                self.use_write_local_cache = [False for i in range(self.num_outputs)]
                self.use_read_share_cache = [False for i in range(self.num_inputs)]
                self.use_read_local_cache = [False for i in range(self.num_inputs)]
                self.write_share_cache = [None for i in range(self.num_outputs)]
                self.write_local_cache = [None for i in range(self.num_outputs)]
                self.read_share_cache = [None for i in range(self.num_inputs)]
                self.read_local_cache = [None for i in range(self.num_inputs)]
                self.org_spatial_iter_var_names = []
                self.org_reduce_iter_var_names = []
                self.iter_var_dict = dict()
                if self.is_compute:
                    for v in op.axis:
                        self.org_spatial_iter_var_names.append(v.var.name)
                        self.iter_var_dict[v.var.name] = v
                    for v in op.reduce_axis:
                        self.org_reduce_iter_var_names.append(v.var.name)
                        self.iter_var_dict[v.var.name] = v
                self.zyx_numbers = [[], [], [], []]
                self.block_binds = [[], [], []]
                self.threads_binds = [[], [], []]
                self.vthread_binds = [[], [], []]
                self.none_binds_order = []
                self.reduce_numbers = [[], []]
                self.iter_var_feature_dict = dict()
                self.visit_feature = []

        self.op_schedule_dict = dict()
        for op in bfs_order:
            self.op_schedule_dict[op] = OpSchedule(op)

        op_feature_dict = get_axis_feature(self.bfs_order)
        for op, feature in op_feature_dict.items():
            self.op_schedule_dict[op].iter_var_feature_dict = feature.iter_var_feature
            self.op_schedule_dict[op].visit_feature = dict(feature.visit_feature)

        self.schedule_diary = None


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
                tmp_dict[vm.name].append((visit_op.name, [iter_var_dom_dict[vm.name], factors[i], vm.factor]))
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
    diary = []
    diary.append(spatial_outer_extents)
    diary.append(spatial_split_factors)
    diary.append(part_one_order)
    diary.append(parallel_decision)
    diary.append(reduce_outer_extents)
    diary.append(reduce_split_factors)
    diary.append(part_two_order)
    diary.append(part_three_order)

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


def graph_schedule_gpu_specific_any(ops, target, model):
    s = Schedule(ops, target)
    op_feature_dict = dict()

    class OpFeature(object):
        def __init__(self):
            self.shape_feature = None
            self.op_feature = None
            self.iter_var_feature = None
            self.visit_feature = None
            self.other_feature = None
            self.whole_feature = None

    # prepare other feature
    pack_other = []
    # pass zero
    for op, op_sch in s.op_schedule_dict.items():
        tmp = []
        tmp += [float(op_sch.is_compute), float(op_sch.has_reduce), float(op_sch.is_output)]
        tmp += [float(op_sch.num_outputs), float(op_sch.num_inputs)]
        pack_other.append(tmp)
    pack_other_vec = torch.FloatTensor(pack_other)
    pack_other_feature = model("other", pack_other_vec)     # use one batch to compute for all ops
    # pass one
    for i, (op, op_sch) in enumerate(s.op_schedule_dict.items()):
        op_feature_dict[op] = OpFeature()
        op_feature_dict[op].other_feature = pack_other_feature[i]

    # total improve
    total_improve = 1

    # pass two
    for op, op_sch in s.op_schedule_dict.items():
        # prepare shape feature
        shape_vec = torch.FloatTensor(op_sch.shape).unsqueeze(1)
        shape_feature = model("shape", shape_vec)
        op_feature_dict[op].shape_feature = shape_feature

        # prepare iter_var feature
        tmp = []
        index = []
        cur = 0
        for iter_var_name in op_sch.org_spatial_iter_var_names:
            fea = op_sch.iter_var_feature_dict[iter_var_name]
            for v in fea:
                tmp.append(v)
                tmp[-1][0] /= BASE
                tmp[-1][1] /= BASE
            cur = cur + len(fea)
            index.append(cur)
        siter_var_feature, sorder, simprove = model("spatial", torch.FloatTensor(tmp), index)
        # the decision of reorder
        for i, part in enumerate(sorder):
            op_sch.zyx_numbers[i] = part
        total_improve = total_improve * (1 + simprove)


        tmp = []
        index = []
        cur = 0
        for iter_var_name in op_sch.org_reduce_iter_var_names:
            fea = op_sch.iter_var_feature_dict[iter_var_name]
            for v in fea:
                tmp.append(v)
                tmp[-1][0] /= BASE
                tmp[-1][1] /= BASE
            cur = cur + len(fea)
            index.append(cur)
        riter_var_feature, rorder, rimprove = model("reduce", torch.FloatTensor(tmp), index, target)
        # the decision of reorder, record number, not axis
        for i, part in enumerate(rorder):
            op_sch.reduce_numbers[i] = part
        total_improve = total_improve * (1 + rimprove)

        op_feature_dict[op].iter_var_feature = torch.cat([siter_var_feature, riter_var_feature])

        # prepare op_feature
        op_feature_dict[op].op_feature = torch.cat(
            [op_feature_dict[op].shape_feature, op_feature_dict[op].iter_var_feature,
             op_feature_dict[op].other_feature])

    num_feature = []
    num_msg = []
    # pass three
    for op, op_sch in s.op_schedule_dict.items():
        # prepare visit_feature
        op_feature_dict[op].visit_feature = model("visit", op_sch, op_feature_dict)

        # prepare whole_feature
        op_feature_dict[op].whole_feature = torch.cat([op_feature_dict[op].op_feature, op_feature_dict[op].visit_feature])

        # make schedule decisions
        if op_sch.able_inline:
            # compute_inline decision
            ret = model("inline", op_feature_dict[op].whole_feature)
            improve = torch.max(ret)
            total_improve = total_improve * (1 + improve)
            choice = torch.argmax(ret)
            if choice == 0:     # inline
                op_sch.compute_inline = True

        if op_sch.need_cache_read:
            # cache_read decision
            ret = model("cache", op_feature_dict[op].whole_feature)
            improve = torch.max(ret)
            total_improve = total_improve * (1 + improve)
            choice = torch.argmax(ret)
            if choice == 0:  # use read cache: local + share
                for i in range(op_sch.num_inputs):
                    op_sch.use_read_local_cache[i] = True
                    op_sch.use_read_share_cache[i] = True
            elif choice == 1:   # use read cache: share
                for i in range(op_sch.num_inputs):
                    op_sch.use_read_share_cache[i] = True

        if op_sch.need_cache_write:
            # cache_write decision
            for i in range(op_sch.num_outputs):
                op_sch.use_write_local_cache[i] = True      # fix decision
            num_feature.append(op_feature_dict[op].whole_feature)
            for name, axis in op_sch.iter_var_dict.items():
                num_msg.append(axis.dom.extent.value / BASE)

    # prepare factors
    bx, by, bz, tx, ty, tz, vtx, vty, vtz, rf, improve = model("factor", num_feature, num_msg)
    total_improve = total_improve * (1 + improve)
    s.block_factors[0] = min(bx, s.block_factors[0])
    s.block_factors[1] = min(by, s.block_factors[1])
    s.block_factors[2] = min(bz, s.block_factors[2])
    s.thread_factors[0] = min(tx, s.thread_factors[0])
    s.thread_factors[1] = min(ty, s.thread_factors[1])
    s.thread_factors[2] = min(tz, s.thread_factors[2])
    s.vthread_factors[0] = min(vtx, s.vthread_factors[0])
    s.vthread_factors[1] = min(vty, s.vthread_factors[1])
    s.vthread_factors[2] = min(vtz, s.vthread_factors[2])

    # real schedule
    sch = tvm.create_schedule(ops)
    diary = []
    if s.need_bind:
        # bind
        tbx = tvm.thread_axis("blockIdx.x")
        tby = tvm.thread_axis("blockIdx.y")
        tbz = tvm.thread_axis("blockIdx.z")
        ttx = tvm.thread_axis("threadIdx.x")
        tty = tvm.thread_axis("threadIdx.y")
        ttz = tvm.thread_axis("threadIdx.z")
        tvx = tvm.thread_axis("vthread", name="vx")
        tvy = tvm.thread_axis("vthread", name="vy")
        tvz = tvm.thread_axis("vthread", name="vz")

        tb_lst = [tbx, tby, tbz]
        tt_lst = [ttx, tty, ttz]
        tv_lst = [tvx, tvy, tvz]
    for op, op_sch in s.op_schedule_dict.items():
        if op_sch.compute_inline:
            # compute_inline
            sch[op].compute_inline()
            diary.append("{} compute_inline".format(op.name))

        for i in range(op_sch.num_inputs):
            # cache_read
            if op_sch.use_read_share_cache[i]:
                op_sch.read_share_cache[i] = sch.cache_read(op_sch.input_tensors[i], "shared", [op])
            if op_sch.use_read_local_cache[i]:
                if op_sch.read_share_cache[i] is not None:
                    op_sch.read_local_cache[i] = sch.cache_read(op_sch.read_share_cache[i], "local", [op])
                else:
                    op_sch.read_local_cache[i] = sch.cache_read(op_sch.input_tensors[i], "local", [op])

        for i in range(op_sch.num_outputs):
            # cache_write
            if op_sch.use_write_local_cache[i]:
                op_sch.write_local_cache[i] = sch.cache_write(op_sch.output_tensors[i], "local")
            if op_sch.use_write_share_cache[i]:
                if op_sch.write_local_cache[i] is not None:
                    op_sch.write_share_cache[i] = sch.cache_write(op_sch.write_local_cache[i], "shared")
                else:
                    op_sch.write_share_cache[i] = sch.cache_write(op_sch.output_tensors[i], "shared")

        if op_sch.is_compute and not op_sch.compute_inline:
            # split, reorder
            x_box, y_box, z_box, r_box = [], [], [], []
            extent = []
            boxes = [z_box, y_box, x_box, r_box]
            for i, part in enumerate(op_sch.zyx_numbers):
                tmp = 1
                for number in part:
                    axis = sch[op].op.axis[number]
                    boxes[i].append(axis)
                    tmp *= axis.dom.extent.value
                extent.append(tmp)
            fused = []
            use_fuse = [False, False, False]
            pre_order = []
            thread_use = [[False, False, False], [False, False, False], [False, False, False]]
            for i in range(3):
                part = boxes[i]
                if len(part) > 1:
                    fused.append(sch[op].fuse(*part))
                    diary.append("{} fuse [{}]".format(op.name, [v.var.name for v in part]))
                    use_fuse[i] = True
                elif len(part) == 1:
                    fused.append(part[0])
                else:
                    fused.append(None)
            for i in range(3):
                if extent[i] > 1:
                    outer, inner = sch[op].split(fused[i], nparts=s.block_factors[i])
                    diary.append("{} split {} nparts={}".format(op.name, fused[i].var.name, s.block_factors[i]))
                    op_sch.block_binds[i].append(outer)
                    thread_use[0][i] = True
                    pre_order.append(outer)
                    if use_fuse[i]:
                        extent[i] = 1       # refuse to further schedule fused axis
                        fused[i] = None
                    else:
                        extent[i] = math.ceil(extent[i] / s.block_factors[i])
                        fused[i] = inner
            for i in range(3):
                if extent[i] > 1:
                    outer, inner = sch[op].split(fused[i], nparts=s.vthread_factors[i])
                    diary.append("{} split {} nparts={}".format(op.name, fused[i].var.name, s.vthread_factors[i]))
                    op_sch.vthread_binds[i].append(outer)
                    thread_use[1][i] = True
                    extent[i] = math.ceil(extent[i] / s.vthread_factors[i])
                    pre_order.append(outer)
                    fused[i] = inner
            for i in range(3):
                if extent[i] > 1:
                    outer, inner = sch[op].split(fused[i], nparts=s.thread_factors[i])
                    diary.append("{} split {} nparts={}".format(op.name, fused[i].var.name, s.thread_factors[i]))
                    op_sch.threads_binds[i].append(outer)
                    thread_use[2][i] = True
                    extent[i] = math.ceil(extent[i] / s.thread_factors[i])
                    pre_order.append(outer)
                    fused[i] = inner
            remain = []
            for i in range(3):
                if extent[i] > 1:
                    remain.append(fused[i])
            reorder = r_box + pre_order + remain        # fixed order decision
            sch[op].reorder(*reorder)
            diary.append("{} reorder [{}]".format(op.name, [v.var.name for v in reorder]))

            a_box, b_box = [], []
            boxes = [a_box, b_box]
            split_op = op
            use_cache_write = False
            count_cache = 0
            cache_buf = []
            for cache in op_sch.write_local_cache:
                if cache is not None:
                    count_cache += 1
                    cache_buf.append(cache)
            if count_cache == 1:        # only do so when single write cache
                cache = cache_buf[0]
                split_op = cache.op
                use_cache_write = True
                sch[cache].compute_at(sch[op], pre_order[-1])   # pre_order should not be empty
                diary.append("{} compute_at {} {}".format(cache.name, op.name, pre_order[-1].var.name))
                for i, part in enumerate(op_sch.reduce_numbers):
                    for number in part:
                        axis = sch[cache].op.reduce_axis[number]
                        boxes[i].append(axis)
                remain.clear()
                for i in range(3):
                    if extent[i] > 1:
                        part = op_sch.zyx_numbers[i]
                        if len(part) == 1:      # no consider for fused axis
                            remain.append(sch[cache].op.axis[part[0]])
            if not use_cache_write:
                for i, part in enumerate(op_sch.reduce_numbers):
                    for number in part:
                        axis = sch[op].op.reduce_axis[number]
                        boxes[i].append(axis)

            aouter, ainner = [], []
            for axis in a_box:
                if axis.dom.extent.value >= rf:
                    outer, inner = sch[split_op].split(axis, factor=rf)
                    diary.append("{} split {} factor={}".format(split_op.name, axis.var.name, rf))
                    aouter.append(outer)
                    ainner.append(inner)
                else:
                    aouter.append(axis)
            post_order = aouter + b_box + ainner + remain
            if post_order:      # post_order may be empty
                sch[split_op].reorder(*post_order)
                diary.append("{} reorder [{}]".format(split_op.name, [v.var.name for v in post_order]))
                diary.append("{} reorder [{}]".format(split_op.name, [v.var.name for v in post_order]))

            # fixed compute_at
            for i, cache in enumerate(op_sch.read_share_cache):
                if cache is not None:
                    if b_box:
                        pos = b_box[-1]
                    elif aouter:
                        pos = aouter[-1]
                    else:
                        pos = None
                    if pos is not None:
                        # schedule load share memory
                        sch[cache].compute_at(sch[split_op], pos)
                        diary.append("{} compute_at {} {}".format(cache.name, split_op.name, pos.var.name))
                        active_axis_name = [[], [], []]
                        for j in range(3):
                            if extent[j] > 1:
                                for number in op_sch.zyx_numbers[j]:
                                    active_axis_name[j].append(op.axis[number])
                        visit_by = s.op_schedule_dict[op_sch.input_tensors[i].op].visit_feature[op]
                        active_dim = []
                        for dim, (d, v) in enumerate(visit_by):
                            flag = -1
                            for i in range(3):
                                for name in active_axis_name[i]:
                                    if name in d:
                                        flag = i
                                        break
                            if flag >= 0:
                                active_dim.append((dim, flag))
                        active_dim_dict = dict(active_dim)
                        split_candidates = []
                        non_split = []
                        split_results = []
                        for i, axis in enumerate(sch[cache].op.axis):
                            if i in active_dim_dict:
                                outer, inner = sch[cache].split(axis, nparts=s.thread_factors[active_dim_dict[i]])
                                diary.append("{} split {} nparts={}".format(cache.name, axis.var.name, s.thread_factors[active_dim_dict[i]]))
                                split_candidates.append(outer)
                                if s.need_bind:
                                    sch[cache].bind(outer, tt_lst[active_dim_dict[i]])
                                    diary.append("{} bind {} {}".format(cache.name, outer.var.name, "threadIdx.{}".format(["x", "y", "z"][active_dim_dict[i]])))
                                split_results.append(inner)
                            else:
                                non_split.append(axis)
                        reorder = split_candidates + non_split + split_results
                        sch[cache].reorder(*reorder)
                        diary.append("{} reorder [{}]".format(cache.name, [v.var.name for v in reorder]))
                    else:
                        sch[cache].compute_inline()
                        diary.append("{} compute_inline".format(cache.name))
            for cache in op_sch.read_local_cache:
                if cache is not None:
                    if ainner:
                        pos = ainner[-1]
                    elif remain:
                        pos = remain[0]
                    else:
                        pos = None
                    if pos is not None:
                        sch[cache].compute_at(sch[split_op], pos)
                        diary.append("{} compute_at {} {}".format(cache.name, split_op.name, pos.var.name))
                    else:
                        sch[cache].compute_inline()
                        diary.append("{} compute_inline".format(cache.name))
            if s.need_bind:
                # bind
                for i in range(3):
                    if thread_use[0][i]:
                        sch[op].bind(op_sch.block_binds[i][0], tb_lst[i])
                        diary.append("{} bind {} {}".format(op.name, op_sch.block_binds[i][0].var.name, "blockIdx.{}".format(["x", "y", "z"][i])))
                    if thread_use[1][i]:
                        sch[op].bind(op_sch.vthread_binds[i][0], tv_lst[i])
                        diary.append("{} bind {} {}".format(op.name, op_sch.vthread_binds[i][0].var.name, "vthreadIdx.{}".format(["x", "y", "z"][i])))
                    if thread_use[2][i]:
                        sch[op].bind(op_sch.threads_binds[i][0], tt_lst[i])
                        diary.append("{} bind {} {}".format(op.name, op_sch.threads_binds[i][0].var.name, "threadIdx.{}".format(["x", "y", "z"][i])))
    s.schedule_diary = diary
    return sch, s, total_improve
