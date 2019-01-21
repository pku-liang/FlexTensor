import torch
import torch.nn as nn
from auto_schedule.config import *


class ComputeAtEngine(nn.Module):
    def __init__(self):
        super(ComputeAtEngine, self).__init__()
        self.l1 = nn.Linear(GROUP_SIZE * MAX_PARENTS * MAX_DIM * MAX_VISITS * 4, 64)
        self.activate = torch.tanh
        self.l2 = nn.Linear(64, 32)
        self.t = nn.Linear(TARGET_MSG_LEN, TARGET_MSG_LEN)
        self.l3 = nn.Linear(32 + TARGET_MSG_LEN, 3)

    def forward(self, feature, target):
        tmp = self.activate(self.l1(feature))
        tmp = self.activate(self.l2(tmp))
        t = self.activate(self.t(target))
        tmp = torch.tanh(self.l3(torch.cat([tmp, t])))
        return tmp


class ComputeInlineEngine(nn.Module):
    def __init__(self):
        super(ComputeInlineEngine, self).__init__()
        self.l1 = nn.Linear(GROUP_SIZE * MAX_PARENTS * MAX_DIM * MAX_VISITS * 4, 64)
        self.activate = torch.tanh
        self.l2 = nn.Linear(64, 32)
        self.t = nn.Linear(TARGET_MSG_LEN, TARGET_MSG_LEN)
        self.l3 = nn.Linear(32 + TARGET_MSG_LEN, 2)

    def forward(self, feature, target):
        tmp = self.activate(self.l1(feature))
        tmp = self.activate(self.l2(tmp))
        t = self.activate(self.t(target))
        tmp = torch.tanh(self.l3(torch.cat([tmp, t])))
        return tmp


class SplitEngine(nn.Module):
    def __init__(self):
        super(SplitEngine, self).__init__()
        self.l1 = nn.Linear(GROUP_SIZE * MAX_PARENTS * MAX_DIM * MAX_VISITS * 4, 64)
        self.activate = torch.tanh
        self.l2 = nn.Linear(64, 32)
        self.t = nn.Linear(TARGET_MSG_LEN, TARGET_MSG_LEN)
        self.l3 = nn.Linear(32 + TARGET_MSG_LEN, len(LOCAL_FACTOR_LST))
        self.l4 = nn.Linear(32 + TARGET_MSG_LEN, len(THREAD_FACTOR_LST))

    def forward(self, feature, target):
        tmp = self.activate(self.l1(feature))
        tmp = self.activate(self.l2(tmp))
        t = self.activate(self.t(target))
        local = torch.tanh(self.l3(torch.cat([tmp, t])))
        parallel = torch.tanh(self.l4(torch.cat([tmp, t])))
        return local, parallel
