import torch
import torch.nn as nn
from auto_schedule.utils import mean, dev, two_factor_split


class TargetEngine(nn.Module):
    def __init__(self, dim, target_dim):
        super(TargetEngine, self).__init__()
        self.target_linear = nn.Linear(target_dim, 128)
        self.feature_linear = nn.Linear(dim, 128)
        self.linear = nn.Linear(256, dim)

    def forward(self, feature, target_feature):
        a = torch.relu(self.feature_linear(feature))
        b = torch.relu(self.target_linear(target_feature))
        c = torch.relu(self.linear(torch.cat([a, b])))
        return c


class ComputeInlineEngine(nn.Module):
    def __init__(self, dim):
        super(ComputeInlineEngine, self).__init__()
        self.linear = nn.Linear(dim, 256)
        self.output = nn.Linear(256, 2)

    def forward(self, feature):
        return torch.tanh(self.output(torch.tanh(self.linear(feature))))


class SplitEngine(nn.Module):
    def __init__(self, dim, iter_var_dim, factor_dim):
        super(SplitEngine, self).__init__()
        self.linear_a = nn.Linear(dim, 128)
        self.linear_b = nn.Linear(factor_dim, 128)
        self.linear_c = nn.Linear(384, 32)
        self.linear_d = nn.Linear(iter_var_dim, 128)
        self.output = nn.Linear(32, 1)

    def forward(self, feature, iter_var_feature, factors):
        a = torch.tanh(self.linear_a(feature))
        b = torch.tanh(self.linear_b(factors))
        d = torch.tanh(self.linear_d(iter_var_feature))
        c = self.linear_c(torch.cat([a, d, b]))
        return torch.tanh(self.output(c))


class ReorderEngine(nn.Module):
    def __init__(self, dim, iter_var_dim):
        super(ReorderEngine, self).__init__()
        self.linear_a = nn.Linear(dim, 128)
        self.linear_b = nn.Linear(iter_var_dim, 128)
        self.linear_c = nn.Linear(256, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, feature, iter_var_feature):
        a = torch.tanh(self.linear_a(feature))
        b = torch.tanh(self.linear_b(iter_var_feature))
        c = self.linear_c(torch.cat([a, b]))
        return torch.tanh(self.output(c))


class ComputeAtEngine(nn.Module):
    def __init__(self, dim):
        super(ComputeAtEngine, self).__init__()
        self.linear = nn.Linear(dim, 256)
        self.output = nn.Linear(256, 2)

    def forward(self, feature):
        a = torch.tanh(self.linear(feature))
        b = torch.tanh(self.output(a))
        return b


class ParallelEngine(nn.Module):
    def __init__(self, dim):
        super(ParallelEngine, self).__init__()
        self.linear = nn.Linear(dim, 256)
        self.output = nn.Linear(256, 2)

    def forward(self, feature):
        a = torch.tanh(self.linear(feature))
        b = torch.tanh(self.output(a))
        return b


class UnrollEngine(nn.Module):
    def __init__(self, dim):
        super(UnrollEngine, self).__init__()
        self.linear = nn.Linear(dim, 256)
        self.output = nn.Linear(256, 2)

    def forward(self, feature):
        a = torch.tanh(self.linear(feature))
        b = torch.tanh(self.output(a))
        return b


class VectorizeEngine(nn.Module):
    def __init__(self, dim):
        super(VectorizeEngine, self).__init__()
        self.linear = nn.Linear(dim, 256)
        self.output = nn.Linear(256, 2)

    def forward(self, feature):
        a = torch.tanh(self.linear(feature))
        b = torch.tanh(self.output(a))
        return b


class CacheEngine(nn.Module):
    def __init__(self, dim):
        super(CacheEngine, self).__init__()
        self.linear = nn.Linear(dim, 256)
        self.output = nn.Linear(256, 2)

    def forward(self, feature):
        return torch.tanh(self.output(torch.tanh(self.linear(feature))))


class Policy(nn.Module):
    def __init__(self, dim):
        super(Policy, self).__init__()
        self.target_engine_dict = dict()
        engines = {
            "target": TargetEngine(dim, 16),
            "compute_inline": ComputeInlineEngine(dim * 2),
            "split": SplitEngine(dim, 10, 2),
            "reorder": ReorderEngine(dim, 10),
            "compute_at": ComputeAtEngine(dim * 2),
            "parallel": ParallelEngine(dim),
            "unroll": UnrollEngine(dim),
            "vectorize": VectorizeEngine(dim),
            "cache": CacheEngine(dim * 2)
        }
        self.engines = engines

    def forward(self, type, record, target, random=False, epsilon=None):
        d = self.engines
        target_feature = torch.FloatTensor(target.feature)
        if type == "compute_inline":
            ret = dict()
            if random and torch.rand(1) < epsilon:
                for op, msg in record.op_msg_dict.items():
                    ret[op] = torch.rand(2)
            else:
                a = d["target"](record.tree_feature[0], target_feature)
                for op, msg in record.op_msg_dict.items():
                    b = d["target"](msg.feature[0], target_feature)
                    res = d["compute_inline"](torch.cat([a, b]))
                    ret[op] = res
        elif type == "split":
            ret = dict()
            if random and torch.rand(1) < epsilon:
                for op, msg in record.op_msg_dict.items():
                    tmp = dict()
                    for iter_var_name, iter_var_msg in msg.iter_var_name_msg_dict.items():
                        knobs = two_factor_split(iter_var_msg.extent)
                        q_lst = []
                        for knob in knobs:
                            q_lst.append(torch.rand(1))
                        tmp[iter_var_name] = (torch.FloatTensor(q_lst), knobs)
                    ret[op] = tmp
            else:
                for op, msg in record.op_msg_dict.items():
                    tmp = dict()
                    a = d["target"](msg.feature[0], target_feature)
                    for iter_var_name, iter_var_msg in msg.iter_var_name_msg_dict.items():
                        b = [iter_var_msg.extent, iter_var_msg.pos]
                        pre_lst = iter_var_msg.visit[:][0]
                        post_lst = iter_var_msg.visit[:][1]
                        b.extend([mean(pre_lst), max(pre_lst), min(pre_lst), dev(pre_lst)])
                        b.extend([mean(post_lst), max(post_lst), min(post_lst), dev(post_lst)])
                        b = torch.FloatTensor(b)
                        knobs = two_factor_split(iter_var_msg.extent)
                        q_lst = []
                        for knob in knobs:
                            q_lst.append(d["split"](a, b, torch.FloatTensor(knob)))
                        tmp[iter_var_name] = [torch.FloatTensor(q_lst), knobs]
                    ret[op] = tmp
        elif type == "reorder":
            ret = dict()
            if random and torch.rand(1) < epsilon:
                for op, msg in record.op_msg_dict.items():
                    tmp = dict()
                    for iter_var_name, iter_var_msg in msg.iter_var_name_msg_dict.items():
                        tmp[iter_var_name] = torch.rand(1)
                    ret[op] = tmp
            else:
                for op, msg in record.op_msg_dict.items():
                    a = d["target"](msg.feature[0], target_feature)
                    tmp = dict()
                    for iter_var_name, iter_var_msg in msg.iter_var_name_msg_dict.items():
                        b = [iter_var_msg.extent, iter_var_msg.pos]
                        pre_lst = [x[0] for x in iter_var_msg.visit]
                        post_lst = [x[1] for x in iter_var_msg.visit]
                        b.extend([mean(pre_lst), max(pre_lst), min(pre_lst), dev(pre_lst)])
                        b.extend([mean(post_lst), max(post_lst), min(post_lst), dev(post_lst)])
                        b = torch.FloatTensor(b)
                        tmp[iter_var_name] = d["reorder"](a, b)
                    ret[op] = tmp
        elif type == "compute_at":
            ret = dict()
            if random and torch.rand(1) < epsilon:
                for op, msg in record.op_msg_dict.items():
                    ret[op] = torch.rand(2)
            else:
                a = d["target"](record.tree_feature[0], target_feature)
                for op, msg in record.op_msg_dict.items():
                    b = d["target"](msg.feature[0], target_feature)
                    res = d["compute_at"](torch.cat([a, b]))
                    ret[op] = res
        elif type == "cache":
            ret = dict()
            if random and torch.rand(1) < epsilon:
                for op, msg in record.op_msg_dict.items():
                    ret[op] = torch.rand(2)
            else:
                a = d["target"](record.tree_feature[0], target_feature)
                for op, msg in record.op_msg_dict.items():
                    b = d["target"](msg.feature[0], target_feature)
                    res = d["cache"](torch.cat([a, b]))
                    ret[op] = res
        else:
            raise ValueError()
        return ret




