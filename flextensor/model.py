import os
import time
import torch
import torch.nn as nn
import numpy as np
import copy
import heapq
import json
import flextensor.space as Space
from flextensor.utils import assert_print


global_walker_judger_model_path_prefix = "walker_judger_model_"
global_walker_judger_data_path_prefix = "walker_judger_data_"
global_performance_judger_path_prefix = "performance_judger_model_"
global_performance_data_path_prefix = "performance_judger_data_"


class Judger(torch.nn.Module):
    def __init__(self, input_len, width, depth, output_len):
        super(Judger, self).__init__()
        assert_print(isinstance(width, int) and width > 0)
        assert_print(isinstance(depth, int) and depth > 1)
        self.net = nn.Sequential()
        self.net.add_module("input", nn.Linear(input_len, width))
        self.net.add_module("input_activate", nn.ReLU())
        for count in range(depth - 2):
            name = "hidden_{}".format(count)
            self.net.add_module(name, nn.Linear(width, width))
            self.net.add_module(name + "_activate", nn.ReLU())
        self.net.add_module("output", nn.Linear(width, output_len))

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class Walker(nn.Module):
    def __init__(self, name, subspace, input_len):
        super(Walker, self).__init__()
        self.subspace = subspace
        self.pre_judger = Judger(input_len, 64, 4, self.subspace.num_direction)
        self.post_judger = Judger(input_len, 64, 4, self.subspace.num_direction)     # post updated
        self.memory = []    # (pre_state, action, post_state, reward)
        self.mem_size = 0
        self.inputs_to_judger = torch.FloatTensor(self.subspace.static_entities)
        self.model_path = global_walker_judger_model_path_prefix + name + ".pkl"
        self.data_path = global_walker_judger_data_path_prefix + name + ".txt"
    
    def random_batch(self, batch_size):
        batch_indices = np.random.randint(0, self.subspace.size, batch_size)
        ret_entities = self._get_batch(batch_indices)
        return ret_entities, batch_indices

    def best_batch(self, batch_size):
        batch_size = min(batch_size, self.subspace.size)
        p_values = self.judger(self.inputs_to_judger).reshape(-1)
        ret_p_values, batch_indices = torch.topk(p_values, batch_size)
        ret_entities = self._get_batch(batch_indices)
        return ret_entities, batch_indices

    def _get_batch(self, batch_indices):
        ret_entities = []
        for index in batch_indices:
            ret_entities.append(self.subspace.get_entity(index))
        return ret_entities

    def record_best(self, best_index, best_value):
        self.memory[best_index] = best_value

    def walk(self, inputs, index_lst, trial, epsilon, gamma):
        q_values_lst = self.pre_judger(torch.FloatTensor(inputs)).detach()
        ret_index_lst = []
        ret_choice_lst = []
        for i, q_values in enumerate(q_values_lst):
            p = np.random.random()
            t = max(epsilon * np.exp(-trial * gamma), 0.1)
            # print("check walker walk", p, t)
            if p <= t:
                choice = np.random.randint(0, self.subspace.num_direction)
            else:
                _, choice = torch.max(q_values, dim=-1)
            direction = self.subspace.get_direction(choice)
            new_index = self.subspace.next_entity(index_lst[i], direction)
            ret_index_lst.append(new_index)
            ret_choice_lst.append(int(choice))
        return ret_index_lst, ret_choice_lst

    def full_walk(self, index):
        new_index_lst = []
        for d in self.subspace.directions:
            new_index = self.subspace.next_entity(index, d)
            new_index_lst.append(new_index)
        return new_index_lst, copy.deepcopy(self.subspace.directions)

    def add_data(self, pre_state, action, post_state, reward):
        self.memory.append((pre_state, action, post_state, reward))
        self.mem_size += 1

    def train(self, lr=0.02, decay=0.9, save=True):
        train_data = self.memory
        data_size = min(self.mem_size, 1000)
        print("train walker data size %d" % data_size)
        np.random.shuffle(train_data)
        optimizer = torch.optim.Adadelta(self.pre_judger.parameters(), lr=lr)
        for ep in range(20):
            loss = 0.0
            for p_data in range(data_size):
                data = train_data[p_data]
                pre_state, action, post_state, reward = data
                y = self.pre_judger(torch.FloatTensor(pre_state))[action]
                t = self.post_judger(torch.FloatTensor(post_state)).detach()
                target = torch.max(t, dim=-1)[0] * decay + reward
                loss = loss + torch.pow(y - target, 2)      # simple MSE
                if (p_data + 1) % 32 == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss = 0.0
            if loss > 0.0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if save:
                self.save_model(self.model_path)
            print("[cur/total]=[%d/%d] | loss=%f" % (ep + 1, 20, float(loss)))        
    
    def save_model(self, model_path):
        self.post_judger.load_state_dict(self.pre_judger.state_dict())
        torch.save(self.pre_judger.state_dict(), model_path)

    def load_model(self, model_path):
        self.pre_judger.load_state_dict(torch.load(model_path))
        self.post_judger.load_state_dict(self.pre_judger.state_dict())

    def save_data(self, data_path):
        with open(data_path, "a") as fout:
            for data in self.memory:
                string = json.dumps(data)
                fout.write(string + "\n")
    
    def load_data(self, data_path):
        with open(data_path, "r") as fin:
            for line in fin:
                data = tuple(json.loads(line))
                self.memory.append(data)
                self.mem_size += 1
    
    def dump_data(self):
        self.save_data(self.data_path)

    def load_or_create_model(self):
        if not os.path.exists(self.model_path):
            self.save_model(self.model_path)
        else:
            self.load_model(self.model_path)

    def prepare_data(self):
        if not os.path.exists(self.data_path):
            raise RuntimeError("Data file not found %s" % self.data_path)
        self.load_data(self.data_path)

    def clear_data(self):
        self.memory.clear()
        self.mem_size = 0


class MemEntity(object):
    def __init__(self, indices, value):
        self.indices = indices
        self.value = value
    
    def __lt__(self, b):
        return self.value < b.value


def rank_loss(y, t):
    assert len(y) == len(t)
    length = len(y)
    loss = 0.0
    for i in range(length):
        for j in range(length):
            tmp = torch.sign(t[i] - t[j]) * (y[i] - y[j])
            loss = loss + torch.log(1 + torch.exp(-tmp))
    return loss


class PerformanceModel(nn.Module):
    def __init__(self, input_len):
        super(PerformanceModel, self).__init__()
        self.input_len = input_len

        self.linear1 = nn.Linear(self.input_len, 32, bias=True)
        # self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=0.01)
        self.activate1 = torch.relu

        self.linear2 = nn.Linear(32, 64, bias=True)
        # self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.01)
        self.activate2 = torch.relu

        self.linear3 = nn.Linear(64, 128, bias=True)
        # self.batch_norm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(p=0.01)
        self.activate3 = torch.relu

        self.linear4 = nn.Linear(128, 64, bias=True)
        # self.batch_norm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(p=0.01)
        self.activate4 = torch.relu

        self.linear5 = nn.Linear(64, 16, bias=True)
        # self.batch_norm5 = nn.BatchNorm1d(128)
        # self.dropout5 = nn.Dropout(p=0.2)
        self.activate5 = torch.relu

        self.linear6 = nn.Linear(16, 1, bias=True)
        self.activate6 = torch.relu

    def forward(self, inputs):
        output1 = self.activate1(self.dropout1(self.linear1(inputs)))
        output2 = self.activate2(self.dropout2(self.linear2(output1)))
        output3 = self.activate3(self.dropout3(self.linear3(output2)))
        output4 = self.activate4(self.dropout4(self.linear4(output3)))
        output5 = self.activate5(self.linear5(output4))
        output6 = self.activate6(self.linear6(output5))
        return output6


class WalkerGroup(object):
    def __init__(self, group_name, space, lr=0.02):
        self.space = space
        self.lr = lr
        self.walkers = dict()
        for name, subspace in self.space.items():
            self.walkers[name] = Walker(group_name + "_" + name, subspace, self.space.dim)
        self.memory = []
        self.mem_size = 0
        self.visit = set()
        self.performance_judger = PerformanceModel(self.space.dim)
        self.perfromance_data = []
        self.model_path = global_performance_judger_path_prefix + group_name + ".pkl"
        self.data_path = global_performance_data_path_prefix + group_name + ".txt"

    def forward(self, batch_size, policy="random"):
        assert_print(policy in ["random", "best"])
        ret = dict()
        for name, walker in self.walkers.items():
            if policy == "random":
                ret_entities, ret_p_values = walker.random_batch(batch_size)
            elif policy == "best":
                ret_entities, ret_p_values = walker.best_batch(batch_size)
            ret[name] = (ret_entities, ret_p_values)
        return ret

    def ever_met(self, indices):
        return str(indices) in self.visit
        
    def record(self, indices, value, random_reject=False, gamma=0.5):
        self.visit.add(str(indices))
        if random_reject:
            p = np.random.random()
            t = np.exp(-gamma * (value - self.top1_value()) / self.top1_value())
            # print("record ", p, t, value, self.top1_value())
            if p <= t:
                heapq.heappush(self.memory, MemEntity(indices, value))
                self.mem_size += 1
        else:
            heapq.heappush(self.memory, MemEntity(indices, value))
            self.mem_size += 1

    def has_more(self):
        return self.mem_size > 0

    def str2dict(self, s):
        def _f(lst):
            return (lst[0], int(lst[1]))
        return dict([_f(x.split(":")) for x in s[1:-1].split(", ")])

    def flatten(self, indices):
        ret = []
        for name, index in indices.items():
            entity = self.walkers[name].subspace.get_entity(index)
            ret.extend(entity)
        return ret
    
    def to_config(self, indices):
        ret = dict()
        for type_key, name_lst in self.space.types.items():
            ret[type_key] = []
            for name in name_lst:
                if name in indices:
                    ret[type_key].append(self.walkers[name].subspace.get_entity(indices[name]))
        return ret

    def record_best(self, best_indice, best_value):
        for name, index in best_indice.items():
            self.walkers[name].record_best(index, best_value)
    
    def top_random(self, gamma=0.5, with_value=False):
        e = np.random.choice(self.memory)
        p = np.random.random()
        t = np.exp(-gamma * (e.value - self.top1_value()) / self.top1_value())
        # print("top random", p, t, e.value, self.top1_value())
        if p <= t:
            if with_value:
                return e.indices, e.value
            return e.indices
        else:
            if with_value:
                return self.top1(), self.top1_value()
            return self.top1()

    def topk(self, k, modify=False, with_value=False):
        if k > self.mem_size:
            k = self.mem_size
        ret = []
        for i in range(k):
            tmp = heapq.heappop(self.memory)
            ret.append(tmp)
        self.mem_size -= k
        if not modify:
            for tmp in ret:
                heapq.heappush(self.memory, tmp)
            self.mem_size += k
        if with_value:
            return [(x.indices, x.value) for x in ret]
        return [x.indices for x in ret]
    
    def top1(self):
        if self.mem_size > 0:
            return self.memory[0].indices
        else:
            return {}
    
    def top1_value(self):
        if self.mem_size > 0:
            return self.memory[0].value
        else:
            return float("inf")
    
    def pop_top(self):
        if self.mem_size > 0:
            self.mem_size -= 1
            return heapq.heappop(self.memory)
        else:
            return MemEntity({}, float("inf"))

    def walk(self, indices_value_lst, trial, epsilon=0.8, gamma=0.01):
        indices_lst, value_lst = [x[0] for x in indices_value_lst], [x[1] for x in indices_value_lst]
        # perform one step walk
        flattened_lst = [self.flatten(indices) for indices in indices_lst]
        ret_from_lst = []
        ret_indices_lst = []
        ret_action_lst = []
        for name in self.walkers.keys():
            index_lst = [indices[name] for indices in indices_lst]
            next_index_lst, direction_lst = self.walkers[name].walk(flattened_lst, index_lst, trial, epsilon, gamma)
            next_indices_lst = copy.deepcopy(indices_lst)
            for i, next_indices in enumerate(next_indices_lst):
                next_indices[name] = next_index_lst[i]
                action =(name, direction_lst[i])
                if not self.ever_met(next_indices):
                    ret_from_lst.append((indices_lst[i], value_lst[i]))
                    ret_indices_lst.append(next_indices)
                    ret_action_lst.append(action)
        return ret_from_lst, ret_indices_lst, ret_action_lst
    
    def full_walk(self, indices, no_repeat=True):
        next_indices_lst = []
        action_lst = []
        for name, index in indices.items():
            next_index_lst, actions = self.walkers[name].full_walk(index)
            for next_index, action in zip(next_index_lst, actions):
                next_indices = copy.deepcopy(indices)
                next_indices[name] = next_index
                if no_repeat:
                    if not self.ever_met(next_indices):
                        next_indices_lst.append(next_indices)
                        action_lst.append((name, action))                            
                else:
                    next_indices_lst.append(next_indices)
                    action_lst.append(action)
        return next_indices_lst, action_lst
    
    def add_data(self, name, pre_state, action, post_state, reward):
        self.walkers[name].add_data(self.flatten(pre_state), action, self.flatten(post_state), reward)

    def train_walkers(self):
        for _, walker in self.walkers.items():
            walker.train(lr=self.lr)
    
    def add_perf_data(self, indices_lst, performance_lst):
        inputs = []
        for indices in indices_lst:
            inputs.append(self.flatten(indices))
        self.perfromance_data.append((inputs, performance_lst))

    def train_on_perf(self, save=True):
        train_data = self.perfromance_data
        data_size = min(len(train_data), 1000)
        print("train data size is %d" % data_size)
        np.random.shuffle(train_data)
        optimizer = torch.optim.Adadelta(self.performance_judger.parameters(), lr=self.lr)
        for ep in range(20):
            loss = 0.0
            full_loss = 0.0
            for p_data in range(data_size):
                x = train_data[p_data][0]
                t = train_data[p_data][1]
                y = self.performance_judger(torch.FloatTensor(x)).reshape(-1)
                tmp_loss = rank_loss(y, torch.FloatTensor(t))
                loss = loss + tmp_loss
                full_loss += tmp_loss.detach()
                if (p_data + 1) % 32 == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss = 0.0
            if loss > 0.0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if save:
                self.save_performance_judger(self.model_path)
            print("[cur/total]=[%d/%d] | loss=%f" % (ep + 1, 20, full_loss))
    
    def test_accuracy(self):
        train_data = self.perfromance_data
        data_size = min(len(train_data), 1000)
        print("test data size is %d" % data_size)
        np.random.shuffle(train_data)
        loss = 0.0
        for p_data in range(data_size):
            x = train_data[p_data][0]
            t = train_data[p_data][1]
            y = self.performance_judger(torch.FloatTensor(x)).reshape(-1)
            tmp_loss = rank_loss(y, torch.FloatTensor(t))
            loss = loss + tmp_loss
        return float(loss.detach()) / data_size

    def query_performance(self, indices_lst):
        inputs = []
        for indices in indices_lst:
            inputs.append(self.flatten(indices))
        # empty inputs
        if not inputs:
            return []
        perf_lst = self.performance_judger(torch.FloatTensor(inputs)).reshape(-1)
        return perf_lst.detach().tolist()

    def load_performance_judger(self, model_path):
        self.performance_judger.load_state_dict(torch.load(model_path))

    def save_performance_judger(self, model_path):
        torch.save(self.performance_judger.state_dict(), model_path)

    def dump_performance_data(self, data_path):
        with open(data_path, "a") as fout:
            for data in self.perfromance_data:
                string = json.dumps(data)
                fout.write(string + "\n")
    
    def load_performance_data(self, data_path):
        with open(data_path, "r") as fin:
            for line in fin:
                data = tuple(json.loads(line))
                self.perfromance_data.append(data)

    def dump_data(self):
        self.dump_performance_data(self.data_path)
        for _, walker in self.walkers.items():
            walker.dump_data()

    def load_walker_model(self):
        for _, walker in self.walkers.items():
            walker.load_or_create_model()

    def load_walker_data(self):
        for _, walker in self.walkers.items():
            walker.prepare_data()

    def prepare_performance_data(self):
        if not os.path.exists(self.data_path):
            raise RuntimeError("Performance data file not found %s" % self.data_path)
        self.load_performance_data(self.data_path)

    def load_or_create_model(self):
        if not os.path.exists(self.model_path):
            self.save_performance_judger(self.model_path)
        else:
            self.load_performance_judger(self.model_path)

    def clear_performance_data(self):
        self.perfromance_data.clear()
    
    def clear_walker_data(self):
        for _, walker in self.walkers.items():
            walker.clear_data()
    
    def clear_data(self):
        self.clear_performance_data()
        self.clear_walker_data
