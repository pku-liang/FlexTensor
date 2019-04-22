import torch
import torch.nn as nn
from auto_schedule.utils import get_factor_lst, fact, comb, gumbel_softmax


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
    def __init__(self, dim, k, device=torch.device("cpu:0")):
        super(Embed, self).__init__()
        assert isinstance(k, int) and k > 1
        self.dim = dim
        self.k = k
        self.device = device

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
        return torch.FloatTensor(ret).to(self.device)


class Embedx(nn.Module):
    def __init__(self, dim, k):
        super(Embedx, self).__init__()
        assert isinstance(k, int) and k > 1
        self.dim = dim
        self.k = k

    def forward(self, candidates):
        ret = []
        for factor_lst in candidates:
            embed_lst = []
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
                embed_lst.extend(tmp)
            ret.append(tuple(embed_lst))
        return torch.FloatTensor(ret)


class SASD(nn.Module):  # Static Axis Split Decision
    def __init__(self, dim, length, hidden, embedding, device=torch.device("cpu:0")):
        super(SASD, self).__init__()
        self.dim = dim
        self.length = length
        self.hidden = hidden
        self.axis_filters = nn.Linear(dim + length, hidden)
        self.factor_filters = nn.Linear(embedding, hidden)
        self.embedding = Embed(embedding, 10, device)
        self.device = device

    def forward(self, lst, shape_dict, extent, sampling):
        batch = []
        for (op_name, contents) in lst:
            shape = shape_dict[op_name]
            assert len(shape) == self.dim and len(contents) == self.length
            batch.append(shape + contents)
        if batch:
            batch_vec = torch.FloatTensor(batch).to(self.device)
            filtered = torch.tanh(self.axis_filters(batch_vec))   # regard results of matmul as filter processed values
        else:
            filtered = torch.zeros([1, self.hidden]).to(self.device)
        factors = get_factor_lst(extent)
        embedded = self.embedding(factors)
        factor_feature = torch.tanh(self.factor_filters(embedded))  # use filters to extract features of factors
        value = torch.softmax(torch.sum(filtered.matmul(factor_feature.transpose(0, 1)), dim=0), dim=-1)
        if sampling:
            value = gumbel_softmax(torch.log(value + 1e-20))
        choice = factors[torch.argmax(value)]
        return value, choice


class SASDx(nn.Module):  # Static Axis Split Decision
    def __init__(self, x, dim, length, hidden, embedding):
        super(SASDx, self).__init__()
        if not isinstance(x, int) or x < 1:
            self.x = 1
            raise RuntimeWarning("split number x should be int and x >= 1")
        else:
            self.x = x
        self.dim = dim
        self.length = length
        self.hidden = hidden
        self.axis_filters = nn.Linear(dim + length, hidden)
        self.factor_filters = nn.Linear(embedding * self.x, hidden)
        self.embedding = Embedx(embedding, 10)

    def forward(self, lst, shape_dict, extent, sampling, fix_vthred=False):
        batch = []
        for (op_name, contents) in lst:
            shape = shape_dict[op_name]
            assert len(shape) == self.dim and len(contents) == self.length
            batch.append(shape + contents)
        if batch:
            batch_vec = torch.FloatTensor(batch)
            filtered = torch.tanh(self.axis_filters(batch_vec))   # regard results of matmul as filter processed values
        else:
            filtered = torch.zeros([1, self.hidden])

        def split_tail(factor_lst, fix=False):
            ret = []
            for tp in factor_lst:
                if len(tp) < 1:
                    raise RuntimeWarning("factor lst is empty")
                elif fix:
                    tail = tp[-1]
                    if tail % 2 == 0:
                        ret.append((*tp[:-1], 2, tail//2))
                    else:
                        ret.append((*tp[:-1], 1, tail))
                else:
                    tail = tp[-1]
                    sub_factor_lst = get_factor_lst(tail)
                    for sub_factor in sub_factor_lst:
                        ret.append((*tp[:-1], tail//sub_factor, sub_factor))
            return ret

        candidates = [(extent,)]
        for i in range(1, self.x):
            if i == 2 and fix_vthred:
                candidates = split_tail(candidates, True)
            else:
                candidates = split_tail(candidates, False)

        embedded = self.embedding(candidates)
        factor_feature = torch.tanh(self.factor_filters(embedded))  # use filters to extract features of factors
        value = torch.sum(filtered.matmul(factor_feature.transpose(0, 1)), dim=0)
        logits = torch.softmax(value, dim=-1)
        if sampling:
            logits = gumbel_softmax(torch.log(logits + 1e-20))
        choice = candidates[torch.argmax(logits)]
        return logits, choice


class RASDx(nn.Module):  # Random Axis Split Decision
    def __init__(self, x):
        super(RASDx, self).__init__()
        if not isinstance(x, int) or x < 1:
            self.x = 1
            raise RuntimeWarning("split number x should be int and x >= 1")
        else:
            self.x = x

    def forward(self, extent):
        def split_tail(factor_lst):
            ret = []
            for tp in factor_lst:
                if len(tp) < 1:
                    raise RuntimeWarning("factor lst is empty")
                else:
                    tail = tp[-1]
                    sub_factor_lst = get_factor_lst(tail)
                    for sub_factor in sub_factor_lst:
                        ret.append((*tp[:-1], tail//sub_factor, sub_factor))
            return ret

        candidates = [(extent,)]
        for i in range(1, self.x):
            candidates = split_tail(candidates)
        vals = torch.rand(len(candidates))
        return vals, candidates[torch.argmax(vals)]


class RASD(nn.Module):
    def __init__(self, device=torch.device("cpu:0")):
        super(RASD, self).__init__()
        self.device = device

    def forward(self, extent):
        factors = get_factor_lst(extent)
        vals = torch.softmax(torch.rand(len(factors)).to(self.device), dim=-1)
        return vals, factors[torch.argmax(vals)]


class SARD(nn.Module):  # Static Axis Reorder Decision
    def __init__(self, dim, num, length, hidden, k, l, outputs, device=torch.device("cpu:0")):
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
        self.device = device

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
                axis_feature = torch.sum(torch.relu(self.axis_filters(torch.FloatTensor(batch).to(self.device))), dim=0)
            else:
                axis_feature = torch.zeros(self.hidden).to(self.device)
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
    def __init__(self, outputs, device=torch.device("cpu:0")):
        super(RARD, self).__init__()
        assert isinstance(outputs, int)
        self.outputs = outputs
        self.device = device

    def forward(self):
        vals = torch.softmax(torch.rand(self.outputs).to(self.device), dim=-1)
        return vals, torch.argmax(vals)


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


class OpScheduleCPUd5(nn.Module):
    def __init__(self, length, hidden_dim, device=torch.device("cpu:0")):
        # dim0: align dimension; dim1: axis_feature_dim
        super(OpScheduleCPUd5, self).__init__()
        self.dim = 5
        self.hidden = hidden_dim
        self.sasd1 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd2 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd3 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd4 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd5 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd_spatial_lst = [self.sasd1, self.sasd2, self.sasd3, self.sasd4, self.sasd5]

        self.sasd6 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd7 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd8 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd9 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd10 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd_reduce_lst = [self.sasd6, self.sasd7, self.sasd8, self.sasd9, self.sasd10]

        self.rasd1 = RASD(device)
        self.rasd2 = RASD(device)
        self.rasd3 = RASD(device)
        self.rasd4 = RASD(device)
        self.rasd5 = RASD(device)
        self.rasd_lst = [self.rasd1, self.rasd2, self.rasd3, self.rasd4, self.rasd5]

        self.sard1 = SARD(self.dim, self.dim, length, hidden_dim, 128, 128, fact(self.dim), device)
        self.sard2 = SARD(self.dim, self.dim, length, hidden_dim, 128, 128, fact(self.dim), device)
        self.sard3 = SARD(self.dim, self.dim*2, length, hidden_dim, 128, 128, comb(self.dim + self.dim, self.dim), device)

        self.rard1 = RARD(fact(self.dim), device)
        self.rard2 = RARD(fact(self.dim), device)
        self.rard3 = RARD(comb(self.dim + self.dim, self.dim), device)

        self.sapd = SARD(self.dim, self.dim, length, hidden_dim, 128, 128, (self.dim + 1) * self.dim // 2, device)

        self.rapd = RARD((self.dim + 1) * self.dim // 2, device)

        self.sard_lst = [self.sard1, self.sard2, self.sard3, self.sapd]
        self.rard_lst = [self.rard1, self.rard2, self.rard3, self.rapd]

    def _split(self, M_lst, R_lst, candidates, extents, feature, shape, random, sampling):
        assert len(candidates) == self.dim
        choice_lst = []
        value_lst = []
        for i in range(self.dim):
            name = candidates[i]
            extent = extents[i]
            lst = feature[name]
            if random:
                value, choice = R_lst[i](extent)
            else:
                value, choice = M_lst[i](lst, shape, extent, sampling)
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
            return self._split(self.sasd_spatial_lst, self.rasd_lst, candidates, extents, feature, shape, random, sampling)

        elif type_key == "parallel":
            # candidates: list; extents: dict; feature: dict; shape: dict
            return self._reorder_or_parallel(3, candidates, extents, feature, shape, random, sampling, mode=0)

        elif type_key == "reduce":
            # candidates: list; extents: list; feature: dict; shape: dict
            return self._split(self.sasd_reduce_lst, self.rasd_lst, candidates, extents, feature, shape, random, sampling)

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


class OpScheduleGPUd5(nn.Module):
    def __init__(self, length, hidden_dim, device=torch.device("cpu:0")):
        # dim0: align dimension; dim1: axis_feature_dim
        super(OpScheduleGPUd5, self).__init__()
        self.dim = 5
        self.hidden = hidden_dim
        self.sasd1 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd2 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd3 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd4 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd5 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd_spatial_out_lst = [self.sasd1, self.sasd2, self.sasd3, self.sasd4, self.sasd5]

        # these parameters are not used currently
        self.sasd6 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd7 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd8 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd9 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd10 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd_spatial_in_lst = [self.sasd1, self.sasd2, self.sasd3, self.sasd4, self.sasd5] # [self.sasd6, self.sasd7, self.sasd8, self.sasd9, self.sasd10]

        self.sasd11 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd12 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd13 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd14 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd15 = SASD(self.dim, length, hidden_dim, self.dim, device)
        self.sasd_reduce_lst = [self.sasd11, self.sasd12, self.sasd13, self.sasd14, self.sasd15]

        self.rasd1 = RASD(device)
        self.rasd2 = RASD(device)
        self.rasd3 = RASD(device)
        self.rasd4 = RASD(device)
        self.rasd5 = RASD(device)
        self.rasd_lst = [self.rasd1, self.rasd2, self.rasd3, self.rasd4, self.rasd5]

        self.sard1 = SARD(self.dim, self.dim, length, hidden_dim, 128, 128, fact(self.dim), device)
        self.sard2 = SARD(self.dim, self.dim, length, hidden_dim, 128, 128, fact(self.dim), device)
        self.sard3 = SARD(self.dim, self.dim*2, length, hidden_dim, 128, 128, comb(self.dim + self.dim, self.dim), device)

        self.rard1 = RARD(fact(self.dim), device)
        self.rard2 = RARD(fact(self.dim), device)
        self.rard3 = RARD(comb(self.dim + self.dim, self.dim), device)

        self.sard_lst = [self.sard1, self.sard2, self.sard3]
        self.rard_lst = [self.rard1, self.rard2, self.rard3]

    def _split(self, M_lst, R_lst, candidates, extents, feature, shape, random, sampling):
        assert len(candidates) == self.dim
        choice_lst = []
        value_lst = []
        for i in range(self.dim):
            name = candidates[i]
            extent = extents[i]
            lst = feature[name]
            if random:
                value, choice = R_lst[i](extent)
            else:
                value, choice = M_lst[i](lst, shape, extent, sampling)
            choice_lst.append(choice)
            value_lst.append(value)
        return choice_lst, value_lst

    def _reorder(self, id, candidates, extents, feature, shape, random, sampling, mode=0):
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
        if type_key == "spatial_one":
            # candidates: list; extents: list; feature: dict; shape: dict
            return self._split(self.sasd_spatial_out_lst, self.rasd_lst, candidates, extents, feature, shape, random, sampling)

        if type_key == "spatial_three":
            # candidates: list; extents: list; feature: dict; shape: dict
            return self._split(self.sasd_spatial_in_lst, self.rasd_lst, candidates, extents, feature, shape, random, sampling)

        elif type_key == "reduce":
            # candidates: list; extents: list; feature: dict; shape: dict
            return self._split(self.sasd_reduce_lst, self.rasd_lst, candidates, extents, feature, shape, random, sampling)

        elif type_key == "reorder_one":
            # candidates: list; extents: dict; feature: dict; shape: dict
            return self._reorder(0, candidates, extents, feature, shape, random, sampling, mode=0)

        elif type_key == "reorder_two":
            # candidates: list; extents: dict; feature: dict; shape: dict
            return self._reorder(1, candidates, extents, feature, shape, random, sampling, mode=0)

        elif type_key == "reorder_three":
            # candidates: list; extents: dict; feature: dict; shape: dict
            return self._reorder(2, candidates, extents, feature, shape, random, sampling, mode=1)

        else:
            raise ValueError("Not support type: {}".format(type_key))


# class OpScheduleGPUd5(nn.Module):
#     def __init__(self, length, hidden_dim, dim=5, level=4):
#         # dim0: align dimension; dim1: axis_feature_dim
#         super(OpScheduleGPUd5, self).__init__()
#         self.dim = dim
#         self.hidden = hidden_dim
#         self.sasd1 = SASDx(level, self.dim, length, hidden_dim, self.dim)
#         self.sasd2 = SASDx(level, self.dim, length, hidden_dim, self.dim)
#         self.sasd3 = SASDx(level, self.dim, length, hidden_dim, self.dim)
#         self.sasd4 = SASDx(level, self.dim, length, hidden_dim, self.dim)
#         self.sasd5 = SASDx(level, self.dim, length, hidden_dim, self.dim)
#         self.sasd_spatial_lst = [self.sasd1, self.sasd2, self.sasd3, self.sasd4, self.sasd5]
#
#         self.sasd11 = SASDx(2, self.dim, length, hidden_dim, self.dim)
#         self.sasd12 = SASDx(2, self.dim, length, hidden_dim, self.dim)
#         self.sasd13 = SASDx(2, self.dim, length, hidden_dim, self.dim)
#         self.sasd14 = SASDx(2, self.dim, length, hidden_dim, self.dim)
#         self.sasd15 = SASDx(2, self.dim, length, hidden_dim, self.dim)
#         self.sasd_reduce_lst = [self.sasd11, self.sasd12, self.sasd13, self.sasd14, self.sasd15]
#
#         self.rasd1 = RASDx(level)
#         self.rasd2 = RASDx(level)
#         self.rasd3 = RASDx(level)
#         self.rasd4 = RASDx(level)
#         self.rasd5 = RASDx(level)
#         self.rasd_spatial_lst = [self.rasd1, self.rasd2, self.rasd3, self.rasd4, self.rasd5]
#
#         self.rasd6 = RASDx(2)
#         self.rasd7 = RASDx(2)
#         self.rasd8 = RASDx(2)
#         self.rasd9 = RASDx(2)
#         self.rasd10 = RASDx(2)
#         self.rasd_reduce_lst = [self.rasd6, self.rasd7, self.rasd8, self.rasd9, self.rasd10]
#
#         self.sard1 = SARD(self.dim, self.dim, length, hidden_dim, 128, 128, fact(self.dim))
#         self.sard2 = SARD(self.dim, self.dim, length, hidden_dim, 128, 128, fact(self.dim))
#         self.sard3 = SARD(self.dim, self.dim*2, length, hidden_dim, 128, 128, comb(self.dim + self.dim, self.dim))
#
#         self.rard1 = RARD(fact(self.dim))
#         self.rard2 = RARD(fact(self.dim))
#         self.rard3 = RARD(comb(self.dim + self.dim, self.dim))
#
#         self.sard_lst = [self.sard1, self.sard2, self.sard3]
#         self.rard_lst = [self.rard1, self.rard2, self.rard3]
#
#     def _split(self, M_lst, R_lst, candidates, extents, feature, shape, random, sampling):
#         assert len(candidates) == self.dim
#         choice_lst = []
#         value_lst = []
#         for i in range(self.dim):
#             name = candidates[i]
#             extent = extents[i]
#             lst = feature[name]
#             if random:
#                 value, choice = R_lst[i](extent)
#             else:
#                 value, choice = M_lst[i](lst, shape, extent, sampling)
#             choice_lst.append(choice)
#             value_lst.append(value)
#         return choice_lst, value_lst
#
#     def _reorder(self, id, candidates, extents, feature, shape, random, sampling, mode=0):
#         # candidates: list; extents: dict; feature: dict; shape: dict
#         # update feature
#         if random:
#             logits, choice = self.rard_lst[id]()
#         else:
#             name_feature_dict = dict()
#             for name, value in extents.items():
#                 lst = feature[name]
#                 name_feature_dict[name] = []
#                 for (op_name, contents) in lst:
#                     tmp = contents.copy()
#                     if mode == 0:
#                         tmp[1] *= (tmp[0] / value)
#                         tmp[2] *= (tmp[0] / value)
#                         tmp[0] = value
#                     elif mode == 1:
#                         tmp[0] = value
#                     else:
#                         raise ValueError("mode should in {0, 1}")
#                     name_feature_dict[name].append((op_name, tmp))
#             lst = []
#             for name in candidates:
#                 lst.append(name_feature_dict[name])
#             logits, choice = self.sard_lst[id](lst, shape, sampling)
#         return choice, logits
#
#     def forward(self, type_key, candidates, extents, feature, shape, random=False, sampling=True):
#         if type_key == "spatial":
#             # candidates: list; extents: list; feature: dict; shape: dict
#             return self._split(self.sasd_spatial_lst, self.rasd_spatial_lst, candidates, extents, feature, shape, random, sampling)
#
#         elif type_key == "reduce":
#             # candidates: list; extents: list; feature: dict; shape: dict
#             return self._split(self.sasd_reduce_lst, self.rasd_reduce_lst, candidates, extents, feature, shape, random, sampling)
#
#         elif type_key == "reorder_one":
#             # candidates: list; extents: dict; feature: dict; shape: dict
#             return self._reorder(0, candidates, extents, feature, shape, random, sampling, mode=0)
#
#         elif type_key == "reorder_two":
#             # candidates: list; extents: dict; feature: dict; shape: dict
#             return self._reorder(1, candidates, extents, feature, shape, random, sampling, mode=0)
#
#         elif type_key == "reorder_three":
#             # candidates: list; extents: dict; feature: dict; shape: dict
#             return self._reorder(2, candidates, extents, feature, shape, random, sampling, mode=1)
#
#         else:
#             raise ValueError("Not support type: {}".format(type_key))




