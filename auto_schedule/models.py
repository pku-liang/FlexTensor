import torch
import torch.nn as nn
import numpy as np
from auto_schedule.utils import gumbel_softmax, get_factor_lst
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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, activate=torch.relu):
        return activate(self.linear3(activate(self.linear2(activate(self.linear1(inputs))))))


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, vec):
        return (torch.sigmoid(self.linear(vec)) * vec).sum(dim=0)


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
            self.other_module = MLP(other_dim, 512, hidden_dim)
            self.shape_module = RNNCell(1, hidden_dim)
            self.spatial_module = MLP(var_dim, 512, hidden_dim)
            self.reduce_module = MLP(var_dim, 512, hidden_dim)
            self.var_module = RNNCell(hidden_dim, hidden_dim)
            self.visit_module_a = RNNCell(hidden_dim, hidden_dim)
            self.visit_module_b = RNNCell(4 * hidden_dim, hidden_dim)
            self.inline_module = MLP(4 * hidden_dim, 512, hidden_dim)
            self.cache_module = MLP(4 * hidden_dim, 512, hidden_dim)
            self.factor_module = MLP(4 * hidden_dim, 512, hidden_dim)
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


class OpScheduleCPU(nn.Module):
    def __init__(self, dim0, dim1, hidden_dim):
        # dim0: align dimension; dim1: axis_feature_dim
        super(OpScheduleCPU, self).__init__()
        self.dim = dim0
        self.hidden = hidden_dim
        self.M1 = MLP(dim0 + dim1, hidden_dim, hidden_dim)  # shape + axis_feature
        self.M2 = MLP(1, hidden_dim, hidden_dim)    # factor
        self.M3 = MLP(hidden_dim * 2, hidden_dim, 1)    # spatial improve value
        self.M4 = MLP(hidden_dim, hidden_dim, 1)  # parallel improve value
        self.M5 = MLP(hidden_dim * 2, hidden_dim, 1)    # reduce improve value
        self.M6 = MLP(hidden_dim, hidden_dim, 1)    # reorder part one
        self.M7 = MLP(hidden_dim, hidden_dim, 1)    # reorder part two
        self.M8 = MLP(hidden_dim, hidden_dim, 1)    # reorder part three
        self.A1 = Attention(hidden_dim)     # axis
        self.A2 = Attention(hidden_dim)     # parallel
        self.A3 = Attention(hidden_dim)     # reorder
        self.L1 = LSTM(hidden_dim, hidden_dim)      # reorder

    def _split(self, M, A, F, J, candidates, extents, feature, shape, device):
        pack_vec = []
        pack_index = []
        cur = 0
        for name in candidates:
            lst = feature[name]
            for (op_name, contents) in lst:
                sh = shape[op_name]
                pack_vec.append(sh + contents)
                cur += 1
            pack_index.append(cur)
        if pack_vec:
            result = M(torch.FloatTensor(pack_vec).to(device))
        else:
            result = torch.FloatTensor([]).to(device)
        pack_vec.clear()
        cur = 0
        for i, end in enumerate(pack_index):
            if cur < end:
                f = A(result[cur:end])
                pack_vec.append(f)
            else:
                pack_vec.append(torch.zeros(self.hidden).to(device))
            cur = end
        if len(pack_vec) != self.dim:
            raise RuntimeError()
        fpack_vec = []
        pack_index.clear()
        cur = 0
        tmp_lst = []
        factor_lst = []
        for i in range(self.dim):
            lst = get_factor_lst(extents[i])
            factor_lst.append(lst)
            cur += len(lst)
            pack_index.append(cur)
            tmp_lst.extend(lst)
        vec = torch.FloatTensor(tmp_lst).view(-1, 1).to(device)
        factor_vec = F(vec)
        cur = 0
        for i in range(self.dim):
            # TODO Need batch concat
            for j in range(cur, pack_index[i]):
                fpack_vec.append(torch.cat([pack_vec[i], factor_vec[j]]))
            cur = pack_index[i]
        result = J(torch.stack(fpack_vec), torch.relu)
        improve_lst = []
        choice_lst = []
        cur = 0
        for i in range(self.dim):
            tmp = result[cur:pack_index[i]].view(-1).unsqueeze(0)
            gs = gumbel_softmax(tmp, device, self.training)
            choice_lst.append(factor_lst[i][torch.argmax(gs)])
            improve_lst.append(torch.sum(tmp * gs))
            cur = pack_index[i]
        improve = sum(improve_lst)
        return choice_lst, improve

    def _reorder(self, M, A, L, J, candidates, extents, feature, shape, device, mode=0):
        # candidates: list; extents: dict; feature: dict; shape: dict
        names = []
        pack_vec = []
        pack_index = []
        cur = 0
        for name, value in extents.items():
            lst = feature[name]
            names.append(name)
            for (op_name, contents) in lst:
                sh = shape[op_name]
                tmp = contents.copy()
                if mode == 0:
                    tmp[1] *= (tmp[0] // value)
                    tmp[2] *= (tmp[0] // value)
                    tmp[0] = value
                elif mode == 1:
                    tmp[0] = value
                else:
                    raise ValueError("mode should in {0, 1}")
                pack_vec.append(sh + tmp)
                cur += 1
            pack_index.append(cur)
        if pack_vec:
            result = M(torch.FloatTensor(pack_vec).to(device))
        else:
            result = torch.FloatTensor([]).to(device)
        pack_vec.clear()
        cur = 0
        for i, end in enumerate(pack_index):
            if cur < end:
                f = A(result[cur:end])
                pack_vec.append(f)
            else:
                pack_vec.append(torch.zeros(self.hidden).to(device))
            cur = end
        vec_dict = dict(zip(names, pack_vec))
        vec_dict["none"] = torch.zeros(self.hidden).to(device)
        candidate_vec = []
        for candidate in candidates:
            tmp = []
            for ele in candidate:
                tmp.append(vec_dict[ele])
            candidate_vec.append(torch.stack(tmp))
        candidate_vec = torch.stack(candidate_vec)
        result = L(candidate_vec)
        result = J(result, torch.relu).view(-1).unsqueeze(0)
        gs = gumbel_softmax(result, device, self.training)
        choice = candidates[torch.argmax(gs)]
        improve = (gs * result).sum()
        return choice, improve

    def forward(self, type_key, candidates, extents, feature, shape, device, random=False):
        if type_key == "spatial":
            # candidates: list; extents: list; feature: dict; shape: dict
            if random:
                final_lst = []
                for i in range(self.dim):
                    f_lst = get_factor_lst(extents[i])
                    final_lst.append(f_lst[np.random.randint(0, len(f_lst))])
                return final_lst, 0
            return self._split(self.M1, self.A1, self.M2, self.M3, candidates, extents, feature, shape, device)

        elif type_key == "parallel":
            # candidates: list; extents: dict; feature: dict; shape: dict
            if random:
                return candidates[np.random.randint(len(candidates))], 0
            return self._reorder(self.M1, self.A2, self.L1, self.M4, candidates, extents, feature, shape, device, mode=0)

        elif type_key == "reduce":
            # candidates: list; extents: list; feature: dict; shape: dict
            if random:
                final_lst = []
                for i in range(self.dim):
                    f_lst = get_factor_lst(extents[i])
                    final_lst.append(f_lst[np.random.randint(0, len(f_lst))])
                return final_lst, 0
            return self._split(self.M1, self.A1, self.M2, self.M5, candidates, extents, feature, shape, device)

        elif type_key == "reorder_one":
            # candidates: list; extents: dict; feature: dict; shape: dict
            if random:
                return candidates[np.random.randint(len(candidates))], 0
            return self._reorder(self.M1, self.A3, self.L1, self.M6, candidates, extents, feature, shape, device, mode=0)

        elif type_key == "reorder_two":
            # candidates: list; extents: dict; feature: dict; shape: dict
            if random:
                return candidates[np.random.randint(len(candidates))], 0
            return self._reorder(self.M1, self.A3, self.L1, self.M7, candidates, extents, feature, shape, device, mode=0)

        elif type_key == "reorder_three":
            # candidates: list; extents: dict; feature: dict; shape: dict
            if random:
                return candidates[np.random.randint(len(candidates))], 0
            return self._reorder(self.M1, self.A3, self.L1, self.M8, candidates, extents, feature, shape, device, mode=1)

        else:
            raise ValueError("Not support type: {}".format(type_key))




