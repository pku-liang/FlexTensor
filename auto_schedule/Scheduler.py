import tvm
import torch
import torch.nn as nn
import numpy as np
import auto_schedule.Environment as Environment
from auto_schedule import Split, Reorder, ComputeInline, ComputeAt
from auto_schedule.utils import find_nearest_factor
from auto_schedule import ComputeInlineEngine, ComputeAtEngine, SplitEngine
from auto_schedule.config import *


class ScheduleMessage(object):
    def __init__(self, env, ops, arg_bufs, target):
        self.env = env
        self.ops = ops
        self.arg_bufs = arg_bufs
        self.target = target
        self.s = tvm.create_schedule(ops)

    def new_schedule(self):
        self.s = tvm.create_schedule(self.ops)

    def schedule_for(self, op, schedule_lst):
        node = self.env.op2node[op]
        for sch in schedule_lst:
            sch.apply(self.s, self.env, node, self.arg_bufs, self.target)

    def reset(self):
        self.new_schedule()
        self.env.reset()


class ScheduleResult(object):
    def __init__(self, sm_lst, op2schedule_lst, op2output_cpat, op2output_split, op2choice_cpat_lst, op2choice_split_lst):
        self.sm_lst = sm_lst
        self.op2schedule_lst = op2schedule_lst
        self.op2output_cpat = op2output_cpat
        self.op2output_split = op2output_split
        self.op2choice_cpat_lst = op2choice_cpat_lst
        self.op2choice_split_lst = op2choice_split_lst


def compute_analysis(env, node, parents):
    pid = []
    p_axes = []
    opindex2pos = {}
    count = 0
    for n in parents:
        opindex = env.op2index[n.op]
        if opindex in opindex2pos:
            continue
        pid.append(opindex)
        p_axes.append([])
        opindex2pos[opindex] = count
        count += 1

    for var_name, features in node.get_axis_features().items():
        for opindex, lst in features.items():
            if opindex in opindex2pos:
                pos = opindex2pos[opindex]
                p_axes[pos].append((node.get_pos(var_name), lst))

    aligned_axes = []
    for axes in p_axes:
        visit_pos2feature = {}
        for axis_pos, lst in axes:
            for feature in lst:
                visit_pos = feature.pos
                tmp = [axis_pos, feature.dom, feature.a, feature.b]
                if visit_pos not in visit_pos2feature:
                    visit_pos2feature[visit_pos] = []
                visit_pos2feature[visit_pos].append(tmp)
        aligned = []
        for i in range(MAX_DIM):
            feature = []
            if i in visit_pos2feature:
                feature = visit_pos2feature[i]
                feature = list(sorted(feature, key=lambda x: x[0], reverse=True))
                feature = feature[:MAX_VISITS]
            if len(feature) < MAX_VISITS:
                res = MAX_VISITS - len(feature)
                for j in range(res):
                    feature.append([0, 0, 0, 0])
            aligned.append(feature)
        aligned_axes.append(aligned)
    if len(aligned_axes) < MAX_PARENTS:
        res = MAX_PARENTS - len(aligned_axes)
        for i in range(res):
            aligned_axes.append([[[0, 0, 0, 0] for k in range(MAX_VISITS)] for j in range(MAX_DIM)])

    return aligned_axes


class BaseScheduler(object):
    def schedule_whole_graph(self, *args, **kwargs):
        raise NotImplementedError("BaseTrainer cannot call schedule_whole_graph")


class NaturalScheduler(BaseScheduler, nn.Module):
    def __init__(self, type='best', epsilon=0.9, batch_size=100):
        super(NaturalScheduler, self).__init__()
        if type not in ['best', 'random']:
            raise ValueError("type should be ['best', 'random']")
        self.compute_at_engine = ComputeAtEngine()
        self.compute_inline_engine = ComputeInlineEngine()
        self.split_engines = []
        for i in range(MAX_LOOP):
            self.split_engines.append(SplitEngine())
        self.type = type
        self._stored_epsilon = epsilon
        self._stored_batch_size = batch_size
        if self.type == 'best':
            self._epsilon = 0.05
            self._batch_size = 1
        else:
            self._epsilon = epsilon
            self._batch_size = batch_size

    def change_mode(self, new_type):
        self.type = new_type
        if self.type == 'best':
            self._epsilon = 0.05
            self._batch_size = 1

    def eval(self):
        self.change_mode('best')

    def train(self):
        self.change_mode('random')
        self._epsilon = self._stored_epsilon
        self._batch_size = self._stored_batch_size

    def pre_phase(self, track, env):
        ret = {}
        for op in track:
            node = env.op2node[op]
            # only consider about compute
            if not isinstance(node, Environment.ComputeVertex):
                continue
            node.reset_all()
            parents = []
            for t in op.input_tensors:
                parents.append(env.op2node[t.op])
            res = compute_analysis(env, node, parents)
            ret[op] = torch.FloatTensor(np.reshape(np.array(res), (-1,)))
        return ret

    def post_phase(self, track, env, ops, arg_bufs, target, op2choice_cpat, op2choice_split):
        op2schedule = {}
        op2compute_at_point = {}
        sm = ScheduleMessage(env, ops, arg_bufs, target)
        for op in track:
            node = env.op2node[op]
            # only consider about compute
            if not isinstance(node, Environment.ComputeVertex):
                continue
            sch_lst = []    # record the schedule for current op
            if op not in op2choice_cpat:
                raise RuntimeError("op not in op2choice_cpat")
            choice_cpat = op2choice_cpat[op]
            need_split = True
            for i, choice in enumerate(choice_cpat):
                # Compute inline
                if choice == 1:
                    if not isinstance(node, Environment.ComputeVertex) or op not in env.down_graph:
                        continue
                    if node.has_reduce:
                        continue
                    sch = ComputeInline()
                    ok, msg = sch.apply(sm.s, sm.env, node, sm.arg_bufs, sm.target)
                    if not ok:
                        raise RuntimeError(msg)
                    sch_lst.append(sch)
                    op2schedule[op] = sch_lst  # store it
                    need_split = False      # Compute inline do not allow split
                # Compute at
                elif choice == 2:
                    if not isinstance(node, Environment.ComputeVertex) or op not in env.down_graph:
                        continue
                    if len(env.down_graph[op]) > 1:
                        continue
                    cop = env.down_graph[op][0]
                    cnode = env.op2node[cop]
                    if not cnode.isolate:
                        continue
                    if cop not in op2compute_at_point:
                        raise RuntimeError("cop not in op2compute_at_point")
                    axis = op2compute_at_point[cop]
                    sch = ComputeAt(axis)
                    ok, msg = sch.apply(sm.s, sm.env, node, sm.arg_bufs, sm.target)
                    if not ok:
                        raise RuntimeError(msg)
                    sch_lst.append(sch)

            if not need_split:
                continue
            if op not in op2choice_split:
                raise RuntimeError("op not in op2choice_split")
            choice_split = op2choice_split[op]
            var_names = list(reversed(node.get_var_names()))
            new_axes = []
            for i, (fl, fp) in enumerate(choice_split):
                fl, fp = LOCAL_FACTOR_LST[fl], THREAD_FACTOR_LST[fp]
                dom = node.get_axis_dom(var_names[i])
                fl = find_nearest_factor(dom, fl)
                fp = find_nearest_factor(fl, fp)
                sch = Split(var_names[i], [fp, fl // fp, dom // fl])
                ok, msg = sch.apply(sm.s, sm.env, node, sm.arg_bufs, sm.target)
                if not ok:
                    raise RuntimeError(msg)
                new_axes.append(sch.get_result_axis_names())
                sch_lst.append(sch)
            parts = [[], [], []]
            for i in range(len(choice_split)):
                parts[0].append(new_axes[i][0])
                parts[1].append(new_axes[i][1])
                parts[2].append(new_axes[i][2])

            op2compute_at_point[op] = parts[0][-1]

            new_order = parts[0] + parts[1] + parts[2]
            sch = Reorder(new_order)
            ok, msg = sch.apply(sm.s, sm.env, node, sm.arg_bufs, sm.target)
            if not ok:
                raise RuntimeError(msg)
            sch_lst.append(sch)

            op2schedule[op] = sch_lst   # store it
        return sm, op2schedule

    def schedule_whole_graph(self, track, env, ops, arg_bufs, target, target_msg):
        op2feature = self.pre_phase(track, env)
        op2output_cpat = {}
        op2output_split = {}
        op2choice_cpat_lst = [{} for i in range(self._batch_size)]
        op2choice_split_lst = [{} for i in range(self._batch_size)]
        for op in track:
            node = env.op2node[op]
            # only consider about compute
            if not isinstance(node, Environment.ComputeVertex):
                continue
            feature = [op2feature[op]]
            count = 0
            for t in op.input_tensors:
                cnode = env.op2node[t.op]
                # only consider about compute
                if not isinstance(cnode, Environment.ComputeVertex):
                    continue
                feature.append(op2feature[t.op])
                count += 1
                if count >= MAX_PARENTS:
                    break
            if count < MAX_PARENTS:
                res = MAX_PARENTS - count
                for i in range(res):
                    feature.append(torch.zeros(MAX_PARENTS * MAX_DIM * MAX_VISITS * 4))
            count = 0
            if op in env.down_graph:
                for cop in env.down_graph[op]:
                    feature.append(op2feature[cop])
                    count += 1
                    if count >= MAX_CHILDREN:
                        break
            if count < MAX_CHILDREN:
                res = MAX_CHILDREN - count
                for i in range(res):
                    feature.append(torch.zeros(MAX_PARENTS * MAX_DIM * MAX_VISITS * 4))
            output_cpat = []
            if op in env.down_graph:
                if len(env.down_graph[op]) == 1:
                    Q_cpat = self.compute_at_engine(torch.cat(feature), target_msg.msg)
                    output_cpat.append(Q_cpat)
                else:
                    Q_cpat = self.compute_inline_engine(torch.cat(feature), target_msg.msg)
                    output_cpat.append(Q_cpat)

            output_split = []
            for i, varname in enumerate(reversed(node.get_var_names())):
                if i >= MAX_LOOP:
                    break
                Q_split_local, Q_split_parallel = self.split_engines[i](torch.cat(feature), target_msg.msg)
                output_split.append([Q_split_local, Q_split_parallel])

            op2output_cpat[op] = output_cpat
            op2output_split[op] = output_split

        sm_lst, op2schedule_lst = [], []

        for bc in range(self._batch_size):
            env.reset()
            for op in track:
                node = env.op2node[op]
                # only consider about compute
                if not isinstance(node, Environment.ComputeVertex):
                    continue
                choice_cpat = []
                choice_split = []
                if self.type == 'best':
                    if op2output_cpat[op]:
                        choice_cpat.append(torch.argmax(op2output_cpat[op][0]))
                    for i, varname in enumerate(reversed(node.get_var_names())):
                        if i >= MAX_LOOP:
                            break
                        choice_split.append((torch.argmax(op2output_split[op][i][0]),
                                             torch.argmax(op2output_split[op][i][1])))
                elif self.type == 'random':
                    if op2output_cpat[op]:
                        choice_cpat.append(int(np.random.random() * (len(op2output_cpat[op][0]) - 1)))
                    for i, varname in enumerate(reversed(node.get_var_names())):
                        if i >= MAX_LOOP:
                            break
                        choice_split.append((int(np.random.random() * (len(op2output_split[op][i][0]) - 1)),
                                             int(np.random.random() * (len(op2output_split[op][i][1]) - 1))))
                op2choice_cpat_lst[bc][op] = choice_cpat
                op2choice_split_lst[bc][op] = choice_split
            sm, op2schedule = self.post_phase(track, env, ops, arg_bufs, target, op2choice_cpat_lst[bc], op2choice_split_lst[bc])
            sm_lst.append(sm)
            op2schedule_lst.append(op2schedule)
        return ScheduleResult(sm_lst, op2schedule_lst, op2output_cpat, op2output_split, op2choice_cpat_lst, op2choice_split_lst)

    def forward(self, track, env, ops, arg_bufs, target, target_msg):
        return self.schedule_whole_graph(track, env, ops, arg_bufs, target, target_msg)
