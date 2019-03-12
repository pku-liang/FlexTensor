import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from auto_schedule.utils import parted_linear, get_factor_lst, fact, comb, dev, gumbel_softmax
from scipy.cluster.vq import kmeans


class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, inputs, hidden, cell):
        catted = torch.cat([inputs, hidden], dim=-1)
        result = self.linear(catted)
        i, j, f, o = torch.split(result, self.hidden_dim, dim=1)
        ai, af, ao = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        aj = torch.tanh(j)
        next_c = ai * aj + cell * af
        next_h = ao * torch.tanh(next_c)
        return next_h, next_c


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.cell = RNNCell(input_dim, hidden_dim)

    def forward(self, inputs):
        seq = []
        tpl = torch.split(inputs, 1, dim=1)
        for ele in tpl:
            seq.append(ele.squeeze())
        hidden = torch.zeros_like(seq[0])
        cell = torch.zeros_like(seq[0])
        for ele in seq:
            hidden, cell = self.cell(ele, hidden, cell)
        return hidden


class Embed(nn.Module):
    def __init__(self, dim, k):
        super(Embed, self).__init__()
        assert isinstance(k, int) and k > 1
        self.dim = dim
        self.k = k

    def forward(self, factor_lst):
        ret = []
        for factor in factor_lst:
            tmp = []
            assert isinstance(factor, int) and factor >= 1
            while factor > 0:
                r = factor % self.k
                factor = factor // self.k
                tmp.append(r)
            cha = max(0, self.dim - len(tmp))
            tmp = tmp[:self.dim]
            tmp = tmp + [0] * cha
            ret.append(tmp)
        return torch.FloatTensor(ret)


class SASD(nn.Module):  # Static Axis Split Decision
    def __init__(self, dim, length, hidden, embedding):
        super(SASD, self).__init__()
        self.dim = dim
        self.length = length
        self.hidden = hidden
        self.axis_filters = nn.Linear(dim + length, hidden)
        self.factor_filters = nn.Linear(embedding, hidden)
        self.embedding = Embed(embedding, 10)

    def forward(self, lst, shape_dict, extent, sampling):
        batch = []
        for (op_name, contents) in lst:
            shape = shape_dict[op_name]
            assert len(shape) == self.dim and len(contents) == self.length
            batch.append(shape + contents)
        if batch:
            batch_vec = torch.FloatTensor(batch)
            filtered = torch.tanh(self.axis_filters(batch_vec))     # regard results of matmul as filter processed values
        else:
            filtered = torch.zeros([1, self.hidden])
        factors = get_factor_lst(extent)
        embedded = self.embedding(factors)
        factor_feature = torch.tanh(self.factor_filters(embedded))  # use filters to extract features of factors
        value = torch.softmax(torch.sum(filtered.matmul(factor_feature.transpose(0, 1)), dim=0), dim=-1)
        if sampling:
            value = gumbel_softmax(torch.log(value + 1e-20))
        choice = factors[torch.argmax(value)]
        return value, choice


class RASD(nn.Module):  # Random Axis Split Decision
    def __init__(self):
        super(RASD, self).__init__()

    def forward(self, extent):
        factors = get_factor_lst(extent)
        return torch.zeros(len(factors)), factors[np.random.randint(0, len(factors))]


class SARD(nn.Module):  # Static Axis Reorder Decision
    def __init__(self, dim, num, length, hidden, k, l, outputs):
        super(SARD, self).__init__()
        self.dim = dim
        self.num = num
        self.length = length
        self.hidden = hidden
        self.axis_filters = nn.Linear(self.dim + self.length, self.hidden)
        self.row_filter = nn.Linear(self.hidden, k)
        self.col_filter = nn.Linear(self.num, l)
        self.flatten = lambda x: x.view(-1)
        self.fc = nn.Linear(k * l, outputs)

    def forward(self, lst, shape_dict, sampling):
        assert len(lst) == self.num
        pack = []
        for sub_lst in lst:
            batch = []
            for (op_name, contents) in sub_lst:
                shape = shape_dict[op_name]
                assert len(contents) == self.length
                batch.append(shape + contents)
            if batch:
                axis_feature = torch.sum(torch.relu(self.axis_filters(torch.FloatTensor(batch))), dim=0)
            else:
                axis_feature = torch.zeros(self.hidden)
            pack.append(axis_feature)
        pack_vec = torch.stack(pack)
        row_filtered = torch.tanh(self.row_filter(pack_vec))
        col_filtered = torch.tanh(self.col_filter(row_filtered.transpose(0, 1)))    # (k x l)
        flatten = self.flatten(col_filtered)
        logits = torch.softmax(self.fc(flatten), dim=-1)
        if sampling:
            logits = gumbel_softmax(torch.log(logits + 1e-20))
        choice = torch.argmax(logits)
        return logits, choice


class RARD(nn.Module):  # Random Axis Reorder Decision
    def __init__(self, outputs):
        super(RARD, self).__init__()
        assert isinstance(outputs, int)
        self.outputs = outputs

    def forward(self):
        return torch.zeros(self.outputs), np.random.randint(0, self.outputs)


class MLP_L2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_L2, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, activate=torch.relu):
        return activate(self.linear2(activate(self.linear1(inputs))))


class MLP_L8(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_L8, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, activate=torch.relu):
        result = activate(self.linear4(activate(self.linear3(activate(self.linear2(activate(self.linear1(inputs))))))))
        result = activate(self.linear8(activate(self.linear7(activate(self.linear6(activate(self.linear5(result))))))))
        return result


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()

    def forward(self, vec):
        return torch.max(vec, dim=0)[0]


class ScheduleModel(nn.Module):
    def __init__(self, block_choice_num, thread_choice_num, random=True, other_dim=None, var_dim=None, hidden_dim=None):
        super(ScheduleModel, self).__init__()
        self.block_choice_num = block_choice_num
        self.thread_choice_num = thread_choice_num
        self.random = random
        if random:
            self.hidden_dim = 1
        else:
            self.cluster_num = 8
            self.hidden_dim = hidden_dim
            self.other_module = MLP_L2(other_dim, 512, hidden_dim)
            self.shape_module = RNNCell(1, hidden_dim)
            self.spatial_module = MLP_L2(var_dim, 512, hidden_dim)
            self.reduce_module = MLP_L2(var_dim, 512, hidden_dim)
            self.var_module = RNNCell(hidden_dim, hidden_dim)
            self.visit_module_a = RNNCell(hidden_dim, hidden_dim)
            self.visit_module_b = RNNCell(4 * hidden_dim, hidden_dim)
            self.inline_module = MLP_L2(4 * hidden_dim, 512, hidden_dim)
            self.cache_module = MLP_L2(4 * hidden_dim, 512, hidden_dim)
            self.factor_module = MLP_L2(4 * hidden_dim, 512, hidden_dim)
            self.number_module = nn.Linear(self.cluster_num, hidden_dim)
            self.judge_module_a = nn.Linear(hidden_dim, 1)
            self.judge_module_b = nn.Linear(hidden_dim, 2)
            self.judge_module_c = nn.Linear(hidden_dim, 3)
            self.judge_module_d = nn.Linear(hidden_dim * 2, block_choice_num * 3 + thread_choice_num * 7)

    def forward(self, type, *args):
        if type == "other":
            packed = args[0]
            if self.random:
                return torch.zeros((packed.shape[0], 1))
            return self.other_module(packed)
        elif type == "shape":
            if self.random:
                return torch.zeros(self.hidden_dim)
            shape_vec = args[0]
            hidden_vec = torch.zeros(self.hidden_dim)
            cell_vec = torch.zeros(self.hidden_dim)
            for part in shape_vec:
                hidden_vec, cell_vec = self.shape_module(part, hidden_vec, cell_vec)
            return hidden_vec
        elif type == "spatial":
            packed = args[0]
            index = args[1]
            num_var = len(index)
            if num_var == 0:
                return torch.zeros(self.hidden_dim), [[], [], [], []], 0
            if self.random:
                result = list(range(num_var))
                np.random.shuffle(result)
                ret_lst = [[], [], [], []]
                p = 0
                for i in range(4):
                    if p < num_var:
                        ret_lst[i].append(result[p])
                        p += 1
                ret_lst[3].extend([x for x in result[4:]])
                return torch.zeros(self.hidden_dim), ret_lst, 0
            feature = self.spatial_module(packed)
            beg = 0
            buffer = []
            for end in index:
                tmp = feature[beg:end]
                max_feature = torch.max(tmp, dim=0)[0]
                buffer.append(max_feature)
                beg = end
            scores = torch.sigmoid(self.judge_module_a(torch.stack(buffer))).squeeze()
            improve = 1
            for value in scores:
                improve = improve * (1 + value)
            improve = improve - 1
            scores = scores.tolist()
            to_sort = [(i, scores[i]) for i in range(num_var)]
            result = sorted(to_sort, key=lambda x: x[1], reverse=True)
            hidden_vec = torch.zeros(self.hidden_dim)
            cell_vec = torch.zeros(self.hidden_dim)
            for vec in buffer:
                hidden_vec, cell_vec = self.var_module(vec, hidden_vec, cell_vec)
            ret_lst = [[], [], [], []]
            p = 0
            for i in range(4):
                if p < num_var:
                    ret_lst[i].append(result[p][0])
                    p += 1
            ret_lst[3].extend([x[0] for x in result[3:]])
            return hidden_vec, ret_lst, improve
        elif type == "reduce":
            packed = args[0]
            index = args[1]
            num_var = len(index)
            if num_var == 0:
                return torch.zeros(self.hidden_dim), [[], []], 0
            if self.random:
                result = list(range(num_var))
                np.random.shuffle(result)
                end = np.random.randint(0, num_var)
                return torch.zeros(self.hidden_dim), [result[:end], result[end:]], 0
            feature = self.reduce_module(packed)
            beg = 0
            buffer = []
            for end in index:
                tmp = feature[beg:end]
                max_feature = torch.max(tmp, dim=0)[0]
                buffer.append(max_feature)
                beg = end
            scores = torch.sigmoid(self.judge_module_a(torch.stack(buffer))).squeeze()
            improve = 1
            for value in scores:
                improve = improve * (1 + value)
            improve = improve - 1
            scores = scores.tolist()
            to_sort = [(i, scores[i]) for i in range(num_var)]
            result = sorted(to_sort, key=lambda x: x[1], reverse=True)
            mean_val = sum(scores) / num_var
            pos = 0
            while pos < num_var and result[pos][0] > mean_val:
                pos += 1
            hidden_vec = torch.zeros(self.hidden_dim)
            cell_vec = torch.zeros(self.hidden_dim)
            for vec in buffer:
                hidden_vec, cell_vec = self.var_module(vec, hidden_vec, cell_vec)
            return hidden_vec, [[x[0] for x in result[:pos]], [x[0] for x in result[pos:]]], improve
        elif type == "visit":
            if self.random:
                return torch.zeros(self.hidden_dim)
            op_sch = args[0]
            op_feature_dict = args[1]
            visit_pack = []
            for op, fea in op_sch.visit_feature.items():
                op_feature = op_feature_dict[op].op_feature
                pack_dim = []
                for (d, v) in fea:
                    pack = []
                    for name, lst in d.items():
                        pack.extend(lst)
                    pack_vec = torch.FloatTensor(pack)
                    max_feature = torch.max(self.spatial_module(pack_vec), dim=0)[0]
                    pack_dim.append(max_feature)
                hidden_vec = torch.zeros(self.hidden_dim)
                cell_vec = torch.zeros(self.hidden_dim)
                for vec in pack_dim:
                    hidden_vec, cell_vec = self.visit_module_a(vec, hidden_vec, cell_vec)
                visit_pack.append(self.visit_module_b(torch.cat([op_feature, hidden_vec])))
            if visit_pack:
                visit_pack_vec = torch.stack(visit_pack)
                visit_feature = torch.max(visit_pack_vec, dim=0)[0]
                return visit_feature
            else:
                return torch.zeros(self.hidden_dim)
        elif type == "inline":
            if self.random:
                return torch.rand(2)
            whole_feature = args[0]
            return torch.sigmoid(self.judge_module_b(self.inline_module(whole_feature)))
        elif type == "cache":
            if self.random:
                return torch.rand(3)
            whole_feature = args[0]
            return torch.sigmoid(self.judge_module_c(self.cache_module(whole_feature)))
        elif type == "factor":
            if self.random:
                bx = np.random.randint(0, self.block_choice_num)
                by = np.random.randint(0, self.block_choice_num)
                bz = np.random.randint(0, self.block_choice_num)
                vtx = np.random.randint(0, self.thread_choice_num)
                vty = np.random.randint(0, self.thread_choice_num)
                vtz = np.random.randint(0, self.thread_choice_num)
                tx = np.random.randint(0, self.thread_choice_num)
                ty = np.random.randint(0, self.thread_choice_num)
                tz = np.random.randint(0, self.thread_choice_num)
                rf = np.random.randint(0, self.thread_choice_num)
                return 2**bx, 2**by, 2**bz, 2**vtx, 2**vty, 2**vtz, 2**tx, 2**ty, 2**tz, 2**rf, 0

            num_feature = args[0]
            num_msg = args[1]
            num_msg = sorted(num_msg)
            length = len(num_msg)
            while length < self.cluster_num:
                num_msg.append(num_msg[-1])
                length += 1
            num_msg = np.array(num_msg, dtype=np.float32).reshape((-1, 1))
            num_msg_feature = kmeans(num_msg, self.cluster_num)[0]
            num_feature_vec = torch.max(self.factor_module(num_feature), dim=0)[0]
            num_msg_feature_vec = torch.relu(self.number_module(num_msg_feature))
            total_feature = torch.cat([num_feature_vec, num_msg_feature_vec])
            scores = torch.sigmoid(self.judge_module_d(total_feature))
            base = self.block_choice_num * 3
            bx_improve, bx_f = torch.max(scores[:self.block_choice_num]), torch.argmax(scores[:self.block_choice_num])
            by_improve, by_f = torch.max(scores[self.block_choice_num:self.block_choice_num * 2]), torch.argmax(
                scores[self.block_choice_num:self.block_choice_num * 2])
            bz_improve, bz_f = torch.max(scores[self.block_choice_num * 2:base]), torch.argmax(
                scores[self.block_choice_num * 2:base])
            vtx_improve, vtx_f = torch.max(scores[base:base + self.thread_choice_num]), torch.argmax(
                scores[base:base + self.thread_choice_num])
            vty_improve, vty_f = torch.max(
                scores[base + self.thread_choice_num:base + self.thread_choice_num * 2]), torch.argmax(
                scores[base + self.thread_choice_num:base + self.thread_choice_num * 2])
            vtz_improve, vtz_f = torch.max(
                scores[base + self.thread_choice_num * 2:base + self.thread_choice_num * 3]), torch.argmax(
                scores[base + self.thread_choice_num * 2:base + self.thread_choice_num * 3])
            tx_improve, tx_f = torch.max(
                scores[base + self.thread_choice_num * 3:base + self.thread_choice_num * 4]), torch.argmax(
                scores[base + self.thread_choice_num * 3:base + self.thread_choice_num * 4])
            ty_improve, ty_f = torch.max(
                scores[base + self.thread_choice_num * 4:base + self.thread_choice_num * 5]), torch.argmax(
                scores[base + self.thread_choice_num * 4:base + self.thread_choice_num * 5])
            tz_improve, tz_f = torch.max(
                scores[base + self.thread_choice_num * 5:base + self.thread_choice_num * 6]), torch.argmax(
                scores[base + self.thread_choice_num * 5:base + self.thread_choice_num * 6])
            rf_improve, r_f = torch.max(
                scores[base + self.thread_choice_num * 6:base + self.thread_choice_num * 7]), torch.argmax(
                scores[base + self.thread_choice_num * 6:base + self.thread_choice_num * 7])
            f_lst = [bx_f, by_f, bz_f, vtx_f, vty_f, vtz_f, tx_f, ty_f, tz_f, r_f]
            v_lst = [2**(x.tolist()) for x in f_lst]
            improve = (1 + bx_improve) * (1 + by_improve) * (1 + bz_improve) * (1 + vtx_improve) * (1 + vtz_improve) * (
                        1 + tx_improve) * (1 + ty_improve) * (1 + tz_improve) * (1 + rf_improve) - 1
            return v_lst[0], v_lst[1], v_lst[2], v_lst[3], v_lst[4], v_lst[5], v_lst[6], v_lst[7], v_lst[8], v_lst[9], improve
        else:
            raise ValueError("not support type:{}".format(type))


class OpScheduleCPUd5(nn.Module):
    def __init__(self, length, hidden_dim):
        # dim0: align dimension; dim1: axis_feature_dim
        super(OpScheduleCPUd5, self).__init__()
        self.dim = 5
        self.hidden = hidden_dim
        self.sasd1 = SASD(self.dim, length, hidden_dim, self.dim)
        self.sasd2 = SASD(self.dim, length, hidden_dim, self.dim)
        self.sasd3 = SASD(self.dim, length, hidden_dim, self.dim)
        self.sasd4 = SASD(self.dim, length, hidden_dim, self.dim)
        self.sasd5 = SASD(self.dim, length, hidden_dim, self.dim)
        self.sasd_lst = [self.sasd1, self.sasd2, self.sasd3, self.sasd4, self.sasd5]

        self.rasd1 = RASD()
        self.rasd2 = RASD()
        self.rasd3 = RASD()
        self.rasd4 = RASD()
        self.rasd5 = RASD()
        self.rasd_lst = [self.rasd1, self.rasd2, self.rasd3, self.rasd4, self.rasd5]

        self.sard1 = SARD(self.dim, self.dim, length, hidden_dim, 128, 128, fact(self.dim))
        self.sard2 = SARD(self.dim, self.dim, length, hidden_dim, 128, 128, fact(self.dim))
        self.sard3 = SARD(self.dim, self.dim*2, length, hidden_dim, 128, 128, comb(self.dim + self.dim, self.dim))

        self.rard1 = RARD(fact(self.dim))
        self.rard2 = RARD(fact(self.dim))
        self.rard3 = RARD(comb(self.dim + self.dim, self.dim))

        self.sapd = SARD(self.dim, self.dim, length, hidden_dim, 128, 128, (self.dim + 1) * self.dim // 2)

        self.rapd = RARD((self.dim + 1) * self.dim // 2)

        self.sard_lst = [self.sard1, self.sard2, self.sard3, self.sapd]
        self.rard_lst = [self.rard1, self.rard2, self.rard3, self.rapd]

    def _split(self, candidates, extents, feature, shape, random, sampling):
        assert len(candidates) == self.dim
        choice_lst = []
        value_lst = []
        for i in range(self.dim):
            name = candidates[i]
            extent = extents[i]
            lst = feature[name]
            if random:
                value, choice = self.rasd_lst[i](extent)
            else:
                value, choice = self.sasd_lst[i](lst, shape, extent, sampling)
            choice_lst.append(choice)
            value_lst.append(value)
        return choice_lst, value_lst

    def _reorder_or_parallel(self, id, candidates, extents, feature, shape, random, sampling, mode=0):
        # candidates: list; extents: dict; feature: dict; shape: dict
        # update feature
        if random:
            logits, choice = self.rard_lst[id]()
        else:
            name_feature_dict = dict()
            for name, value in extents.items():
                lst = feature[name]
                name_feature_dict[name] = []
                for (op_name, contents) in lst:
                    tmp = contents.copy()
                    if mode == 0:
                        tmp[1] *= (tmp[0] / value)
                        tmp[2] *= (tmp[0] / value)
                        tmp[0] = value
                    elif mode == 1:
                        tmp[0] = value
                    else:
                        raise ValueError("mode should in {0, 1}")
                    name_feature_dict[name].append((op_name, tmp))
            lst = []
            for name in candidates:
                lst.append(name_feature_dict[name])
            logits, choice = self.sard_lst[id](lst, shape, sampling)
        return choice, logits

    def forward(self, type_key, candidates, extents, feature, shape, random=False, sampling=True):
        if type_key == "spatial":
            # candidates: list; extents: list; feature: dict; shape: dict
            return self._split(candidates, extents, feature, shape, random, sampling)

        elif type_key == "parallel":
            # candidates: list; extents: dict; feature: dict; shape: dict
            return self._reorder_or_parallel(3, candidates, extents, feature, shape, random, sampling, mode=0)

        elif type_key == "reduce":
            # candidates: list; extents: list; feature: dict; shape: dict
            return self._split(candidates, extents, feature, shape, random, sampling)

        elif type_key == "reorder_one":
            # candidates: list; extents: dict; feature: dict; shape: dict
            return self._reorder_or_parallel(0, candidates, extents, feature, shape, random, sampling, mode=0)

        elif type_key == "reorder_two":
            # candidates: list; extents: dict; feature: dict; shape: dict
            return self._reorder_or_parallel(1, candidates, extents, feature, shape, random, sampling, mode=0)

        elif type_key == "reorder_three":
            # candidates: list; extents: dict; feature: dict; shape: dict
            return self._reorder_or_parallel(2, candidates, extents, feature, shape, random, sampling, mode=1)

        else:
            raise ValueError("Not support type: {}".format(type_key))




