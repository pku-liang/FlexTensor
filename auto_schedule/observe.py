import tvm
import torch
import torch.nn as nn


NODE_LIST = [tvm.stmt.LetStmt,
             tvm.stmt.AssertStmt,
             tvm.stmt.ProducerConsumer,
             tvm.stmt.For,
             tvm.stmt.Store,
             tvm.stmt.Provide,
             tvm.stmt.Allocate,
             tvm.stmt.AttrStmt,
             tvm.stmt.Free,
             tvm.stmt.Realize,
             tvm.stmt.Block,
             tvm.stmt.IfThenElse,
             tvm.stmt.Evaluate,
             tvm.stmt.Prefetch,
             tvm.expr.Var,
             tvm.expr.Reduce,
             tvm.expr.FloatImm,
             tvm.expr.IntImm,
             tvm.expr.UIntImm,
             tvm.expr.StringImm,
             tvm.expr.Cast,
             tvm.expr.Add,
             tvm.expr.Sub,
             tvm.expr.Mul,
             tvm.expr.Div,
             tvm.expr.Mod,
             tvm.expr.Min,
             tvm.expr.Max,
             tvm.expr.EQ,
             tvm.expr.NE,
             tvm.expr.LT,
             tvm.expr.LE,
             tvm.expr.GT,
             tvm.expr.GE,
             tvm.expr.And,
             tvm.expr.Or,
             tvm.expr.Not,
             tvm.expr.Select,
             tvm.expr.Load,
             tvm.expr.Ramp,
             tvm.expr.Broadcast,
             tvm.expr.Shuffle,
             tvm.expr.Call,
             tvm.expr.Let]


NODE_DICT = dict()
for i, node in enumerate(NODE_LIST):
    NODE_DICT[node] = i


def prepare_embedding(dim):
    length = len(NODE_LIST)
    embedding_layer = [torch.rand(dim, requires_grad=True) for i in range(length)]
    return embedding_layer


class ChildSumCell(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(ChildSumCell, self).__init__()
        self.U = nn.Linear(input_dim, 4 * hidden_dim)
        self.W = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, input_lst):
        if len(input_lst) == 0:
            return torch.tanh(torch.stack(torch.split(self.U(x), self.hidden_dim, dim=1)).squeeze().sum(dim=0)), torch.zeros(self.hidden_dim)
        hs, cs = [], []
        for input_t in input_lst:
            hs.append(input_t[0])
            cs.append(input_t[1])
        x_U = self.U(x).squeeze()    # 4 * hidden_dim
        hs = torch.stack(hs)
        cs = torch.stack(cs)
        h_W = self.W(hs)        # N x 4 * hidden_dim
        linear = x_U + h_W      # N x 4 * hidden_dim
        i, j, o, f = torch.split(linear, self.hidden_dim, dim=1)
        ai = torch.sigmoid(i.sum(dim=0))    # hidden_dim
        aj = torch.tanh(j.sum(dim=0))
        ao = torch.sigmoid(o.sum(dim=0))
        af = torch.sigmoid(f)               # N * hidden_dim
        remain = af.mul(cs).sum(dim=0)
        increase = ai.mul(aj)
        next_c = (remain + increase).squeeze()
        next_h = ao.mul(torch.tanh(next_c)).squeeze()
        return (next_h, next_c)


class NaryCell(nn.Module):
    def __init__(self, hidden_dim, input_dim, N):
        super(NaryCell, self).__init__()
        self.U = nn.Parameter(torch.rand((input_dim, 4 * hidden_dim), requires_grad=True))
        self.W = nn.Parameter(torch.rand((N, hidden_dim, 3 * hidden_dim), requires_grad=True))
        self.Wf = nn.Parameter(torch.rand((N, hidden_dim, hidden_dim * N), requires_grad=True))
        self.b = nn.Parameter(torch.rand(4 * hidden_dim))
        self.N = N
        self.hidden_dim = hidden_dim

    def forward(self, x, input_lst):
        assert len(input_lst) == self.N
        hs, cs = [], []
        for input_t in input_lst:
            hs.append(input_t[0])
            cs.append(input_t[1])
        cs = torch.stack(cs)
        stacked = torch.stack(hs).unsqueeze(1)     # N x 1 x hidden_dim
        x_U = x.matmul(self.U).squeeze()                    # 4 * hidden_dim
        h_W = stacked.matmul(self.W).sum(dim=0).squeeze()   # 3 * hidden_dim
        f_W_tuple = torch.split(stacked.matmul(self.Wf).sum(dim=0).squeeze(), self.hidden_dim, dim=0)
        f_W = torch.stack(f_W_tuple)        # N x hidden_dim
        linear = x_U[:3 * self.hidden_dim] + h_W + self.b[:3 * self.hidden_dim]
        i, j, o = torch.split(linear, self.hidden_dim, dim=0)    # hidden_dim
        ai = torch.sigmoid(i)
        aj = torch.tanh(j)
        ao = torch.sigmoid(o)
        f_linear = x_U[3 * self.hidden_dim:] + f_W + self.b[3 * self.hidden_dim:]
        af = torch.sigmoid(f_linear)        # N x hidden_dim
        remain = af.mul(cs).sum(dim=0)      # hidden_dim
        increase = ai.mul(aj)               # hidden_dim
        next_c = remain + increase
        next_h = ao.mul(torch.tanh(next_c))
        return (next_h, next_c)


class Observer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Observer, self).__init__()
        embed_num = len(NODE_LIST)
        self.embedding = nn.Embedding(embed_num, embedding_dim)
        self.cell_table = {
            tvm.stmt.LetStmt: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.stmt.AssertStmt: NaryCell(hidden_dim, embedding_dim, 3),
            tvm.stmt.ProducerConsumer: NaryCell(hidden_dim, embedding_dim, 1),
            tvm.stmt.For: NaryCell(hidden_dim, embedding_dim, 1),
            tvm.stmt.Store: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.stmt.Provide: ChildSumCell(hidden_dim, embedding_dim),
            tvm.stmt.Allocate: NaryCell(hidden_dim, embedding_dim, 1),
            tvm.stmt.AttrStmt: NaryCell(hidden_dim, embedding_dim, 1),
            tvm.stmt.Block: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.stmt.IfThenElse: NaryCell(hidden_dim, embedding_dim, 3),
            tvm.stmt.Evaluate: NaryCell(hidden_dim, embedding_dim, 1),
            tvm.expr.Reduce: ChildSumCell(hidden_dim, embedding_dim),
            tvm.expr.Cast: NaryCell(hidden_dim, embedding_dim, 1),
            tvm.expr.Var: ChildSumCell(hidden_dim, embedding_dim),
            tvm.expr.FloatImm: ChildSumCell(hidden_dim, embedding_dim),
            tvm.expr.IntImm: ChildSumCell(hidden_dim, embedding_dim),
            tvm.expr.UIntImm: ChildSumCell(hidden_dim, embedding_dim),
            tvm.expr.StringImm: ChildSumCell(hidden_dim, embedding_dim),
            tvm.expr.Add: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.Sub: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.Mul: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.Div: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.Mod: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.Min: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.Max: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.EQ: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.NE: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.LT: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.LE: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.GT: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.GE: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.And: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.Or: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.Not: NaryCell(hidden_dim, embedding_dim, 1),
            tvm.expr.Select: NaryCell(hidden_dim, embedding_dim, 3),
            tvm.expr.Load: NaryCell(hidden_dim, embedding_dim, 2),
            tvm.expr.Ramp: NaryCell(hidden_dim, embedding_dim, 1),
            tvm.expr.Broadcast: NaryCell(hidden_dim, embedding_dim, 1),
            tvm.expr.Call: ChildSumCell(hidden_dim, embedding_dim),
            tvm.expr.Let: NaryCell(hidden_dim, embedding_dim, 2)
        }

    def forward(self, node, input_lst):
        index = NODE_DICT[type(node)]
        x = self.embedding(torch.LongTensor([index]))
        func = self.cell_table[type(node)]
        return func(x, input_lst)