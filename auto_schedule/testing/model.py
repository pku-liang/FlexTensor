import time
import torch
import torch.nn as nn
import numpy as np
from auto_schedule.utils import assert_print

class Judger(torch.nn.Module):
    def __init__(self, input_len, width, depth):
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
        self.net.add_module("output", nn.Linear(width, 1))

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class Walker(object):
    def __init__(self, subspace, width=5, depth=5, scope=10, decay=0.7):
        self.subspace = subspace
        self.untouched = list(range(self.subspace.begin, self.subspace.end))
        self.untouched_len = self.subspace.end - self.subspace.begin
        self.memory = dict()
        self.input_len = len(self.subspace.get_entity(0))
        self.judger = Judger(self.input_len, width, depth)
        for p in self.judger.parameters():
            torch.nn.init.uniform_(p)
        self.scope = scope
        self.decay = decay
        # initial random proposals
        self.next_proposal = []
        self.design_proposal(policy="random")

    def propose(self):
        ret = self.next_proposal.copy()
        for (index, value) in ret:
            self.memory[index] = value
        # next proposal
        self.next_proposal = []
        self.design_proposal(policy="best")
        return ret

    def reward(self, feedback):
        for (index, value) in feedback:
            self.memory[index] = (1 - self.decay) * self.memory[index] + self.decay * value
    
    def self_correct(self, strength=10, lr=0.02, verbose=False):
        train_data = list(self.memory.items())
        np.random.shuffle(train_data)
        p = 0
        end = len(train_data)
        optimizer = torch.optim.Adadelta(self.judger.parameters(), lr=lr)
        beg_time = time.time()
        for epoch in range(strength):
            for batch in range(len(self.memory) // self.scope):
                inputs = []
                target = []
                for i in range(self.scope):
                    inputs.append(self.subspace.get_entity(train_data[p][0]))
                    target.append([train_data[p][1]])
                    p = (p + 1) % end
                inputs_torch = torch.FloatTensor(inputs)
                target_torch = torch.FloatTensor(target)
                optimizer.zero_grad()
                y = self.judger(inputs_torch)
                loss = nn.functional.mse_loss(y, target_torch)
                loss.backward()
                optimizer.step()
                if verbose and batch == len(self.memory) // self.scope - 1:
                    print("[epoch/total]=[%d/%d] | [loss=%f] | [use time=.4%fs]"%(
                        epoch, strength, float(loss.detach()), time.time()-beg_time))

    def has_proposal(self):
        return self.untouched_len > 0

    def design_proposal(self, policy="random"):
        assert_print(policy in ["random", "best"])
        count = len(self.next_proposal)
        scope = min(self.scope, self.untouched_len)
        while count < scope:
            if policy == "random":
                next_p = np.random.randint(0, self.untouched_len)
            elif policy == "best":
                indices = np.random.randint(0, self.untouched_len, scope)
                candidates = [self.subspace.get_entity(ind) for ind in indices]
                input_torch = torch.FloatTensor(candidates)
                y = self.judger(input_torch).tolist()
                choice = np.argmax(y)
                next_p = indices[choice]
            else:
                raise RuntimeError()
            index = self.untouched[next_p]
            entity = self.subspace.get_entity(index)
            value = float(self.judger(torch.FloatTensor(entity)).detach())
            self.next_proposal.append((index, value))
            count += 1
            del self.untouched[next_p]
            self.untouched_len -= 1
        self.next_proposal.sort(key=lambda x: x[1], reverse=True)