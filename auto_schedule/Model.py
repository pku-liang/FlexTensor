import torch
import torch.nn as nn
from torch.nn import Parameter as P


class NodeCell(nn.Module):
    """
    A tree GRU cell
    """
    def __init__(self, feature_dim, hidden):
        super(NodeCell, self).__init__()
        self._hidden = hidden
        self._W = nn.Linear(feature_dim + hidden, hidden)
        self._U = nn.Linear(feature_dim + hidden, hidden)
        self._trans = nn.Linear(feature_dim + hidden, hidden)

    def forward(self, x, children_hs):
        num = len(children_hs)
        h = torch.sum(torch.stack(children_hs), dim=0) / num
        r = torch.sigmoid(self._W(torch.cat([x, h])))
        z = torch.sigmoid(self._U(torch.cat([x, h])))
        trans = [torch.tanh(self._trans(torch.cat([x, r * h_]))) for h_ in children_hs]
        next_h = torch.sum(torch.stack([(1 - z) * children_hs[i] + z * trans[i] for i in range(num)]), dim=0) / num
        return next_h


class AxisCell(nn.Module):
    """
    A plain GRU cell
    """
    def __init__(self, feature_dim, hidden):
        super(AxisCell, self).__init__()
        self._hidden = hidden
        self._Linear = nn.Linear(feature_dim + hidden, hidden * 2)
        self._trans = nn.Linear(feature_dim + hidden, hidden)

    def forward(self, x, h):
        pos1 = self._hidden
        activated = torch.sigmoid(self._Linear(torch.cat([x, h])))
        r = activated[:pos1]
        z = activated[pos1:]
        trans = torch.tanh(self._trans(torch.cat([x, r * h])))
        next_h = (1 - z) * h + z * trans
        return next_h


class Agent(nn.Module):
    """
    The agent in reinforcement learning
    """
    def __init__(self, Node_feature_dim, Axis_feature_dim, hidden, attention=False, possible_split_parts=None,
                 reorder_len=4):
        super(Agent, self).__init__()
        # All parameters are here, but not all initialized here
        self._hidden = hidden

        self._node_cell = NodeCell(Node_feature_dim, hidden)
        self._axis_cell = AxisCell(Axis_feature_dim, hidden)
        self._reduce_axis_cell = AxisCell(Axis_feature_dim, hidden)
        self._cell_1 = nn.Linear(Node_feature_dim + Axis_feature_dim, hidden)
        self._cell_2 = NodeCell(Node_feature_dim, hidden)
        # whether to use attention
        if attention:
            self._attention = P(torch.rand([2 * hidden, hidden], requires_grad=True))
        self._with_attention = attention

        # useful states
        self._up_states = {}
        self._down_states = {}
        self._last_h = torch.ones(hidden)
        # use to decide how many parts Split Schedule can split
        self._possible_split_parts = possible_split_parts if possible_split_parts else [2, 3, 4]

        # a layer before deciders
        self._trans = nn.Linear(hidden * 2, hidden)
        # a decider whether ot compute inline, [inline, not_inline]
        self._inline_decider = nn.Linear(hidden, 2)
        # a decider whether and where to compute at, [at, not_at, where]
        self._at_decider = nn.Linear(hidden, 3)
        # a decider whether and how to split an axis, [p1, p2...]
        self._split_decider = nn.Linear(hidden * 2, len(self._possible_split_parts))
        # a decider whether to fuse with the latter axis, [fuse, not_fuse]
        self._fuse_decider = nn.Linear(hidden + Axis_feature_dim, 2)
        # a decider whether to reorder the axes, [weight]
        self._reorder_decider_1 = nn.Linear(hidden * 2, hidden)
        self._reorder_decider_2 = nn.Linear(hidden, 1)
        # a decider whether to parallel the axis, [parallel, not_parallel]
        self._parallel_decider = nn.Linear(hidden + Axis_feature_dim, 2)
        # a decider whether to unroll the axis, [unroll, not_unroll]
        self._unroll_decider = nn.Linear(hidden + Axis_feature_dim, 2)
        # a decider whether to vectorize the axis, [vectorize, not_vectorize]
        self._vectorize_decider = nn.Linear(hidden + Axis_feature_dim, 2)

        # use to mark whether reset
        self._new = False
        self._first_call = True

    def forward(self, op, down_graph, op2node, index2op, op2index, issue, others=None):
        if self._first_call and not self._new:
            Warning("You forget to reset model before using")
        self._first_call = False
        # if self._with_attention:
        #     attention_state = self._get_attention(self._last_h)
        self._update_region_feature(op, down_graph, op2node, op2index)
        state = torch.cat([self._up_states[op], self._down_states[op]])
        self._last_h = torch.relu(self._trans(state))
        if issue == 'none':
            # no schedule
            res = None
        elif issue == 'inline':
            # decide whether to compute inline
            res = torch.softmax(self._inline_decider(self._last_h), dim=0)
        elif issue == 'compute_at':
            # decide wether and where to compute at
            res = torch.softmax(self._at_decider(self._last_h), dim=0)
        elif issue == 'split':
            # for split, others is the features of the axis to split
            tmp = self._pack_axis_feature(op, others, op2node, index2op)
            res = torch.softmax(self._split_decider(torch.cat([self._last_h, tmp])), dim=0)
        elif issue == 'fuse':
            # for fuse, others is the feature of the axis to fuse
            res = torch.softmax(self._fuse_decider(torch.cat([self._last_h, others])), dim=0)
        elif issue == 'reorder':
            # for reorder, others is the list of features of axes to reorder
            tmp_1 = self._pack_node_axis_feature(op, others, op2node, index2op)
            tmp_2 = self._reorder_decider_1(torch.cat([self._last_h, tmp_1]))
            res = self._reorder_decider_2(tmp_2)
        elif issue == 'parallel':
            # for parallel, others is the feature of the axis to parallel
            res = torch.softmax(self._parallel_decider(torch.cat([self._last_h, others])), dim=0)
        elif issue == 'unroll':
            # for unroll, others is the feature of the axis to unroll
            res = torch.softmax(self._unroll_decider(torch.cat([self._last_h, others])), dim=0)
        elif issue == 'vectorize':
            # for vectorize, others is the feature of the axis to vectorize
            res = torch.softmax(self._vectorize_decider(torch.cat([self._last_h, others])), dim=0)
        else:
            raise NotImplementedError("No such schedule decider: {}".format(issue))
        return res

    def get_possible_split_list(self):
        return self._possible_split_parts

    def reset(self, src, end, down_graph, op2node, op2index):

        self._up_states = {}
        self._down_states = {}
        self._last_h = torch.ones(self._hidden)
        self._get_whole_graph_feature(src, end, down_graph, op2node, op2index)
        self._new = True

    def inspect(self):
        ret = ""
        ret += "Inspect temporary states of model:\n"
        ret += "up_states:\n"
        for op, state in self._up_states.items():
            ret += "    {}->{}\n".format(op, state)
        ret += "down_states:\n"
        for op, state in self._down_states.items():
            ret += "    {}->{}\n".format(op, state)
        ret += "last_h:\n"
        ret += "    {}\n".format(self._last_h)
        ret += "*******************************************\n"

    def _get_attention(self, h):
        cache_attentions = {}
        total = torch.tensor(0.0)
        for op, up_state in self._up_states.items():
            down_state = self._down_states[op]
            state = torch.cat([down_state, up_state])
            cache_attentions[op] = state.matmul(self._attention).matmul(h)
            total += cache_attentions[op]
        attention_state = torch.zeros(self._hidden * 2)
        for op, val in cache_attentions.items():
            tmp = val / total
            state = torch.cat([self._down_states[op], self._up_states[op]])
            attention_state += tmp * state
        return attention_state

    def _get_down_graph_feature(self, op, op2node, op2index):
        if op is None:
            return
        if op in self._down_states:
            return
        # current node
        node = op2node[op]
        if not op.input_tensors:
            self._down_states[op] = self._node_cell(node.get_feature(), [torch.zeros(self._hidden)])
            return
        axis_cache = []
        for p in op.input_tensors:
            p_op = p.op
            # parent node
            p_node = op2node[p_op]
            # parent index
            p_index = op2index[p_op]
            self._get_down_graph_feature(p_op, op2node, op2index)
            # if input_tensors is not empty, must be compute node
            axis_count = len(op.axis)
            features = node.get_axis_features()
            for var_name, feature in features.items():
                if p_index in feature:
                    for f in feature[p_index]:
                        # f[1] indicates which axis in op visits p_op
                        if f[1] >= axis_count:
                            axis_cache.append(self._reduce_axis_cell(f, self._down_states[p_op]))
                        else:
                            axis_cache.append(self._axis_cell(f, self._down_states[p_op]))
        self._down_states[op] = self._node_cell(node.get_feature(), axis_cache)

    def _get_up_graph_feature(self, op, down_graph, op2node, op2index):
        if op is None:
            return
        if op in self._up_states:
            return
        node = op2node[op]
        index = op2index[op]
        if op not in down_graph:
            self._up_states[op] = self._node_cell(node.get_feature(), [torch.zeros(self._hidden)])
            return
        axis_cache = []
        for c_op in down_graph[op]:
            # child node
            c_node = op2node[c_op]
            self._get_up_graph_feature(c_op, down_graph, op2node, op2index)
            # c_op must be compute operation
            axis_count = len(c_op.axis)
            features = c_node.get_axis_features()
            for var_name, feature in features.items():
                if index in feature:
                    for f in feature[index]:
                        if f[1] >= axis_count:
                            axis_cache.append(self._reduce_axis_cell(f, self._up_states[c_op]))
                        else:
                            axis_cache.append(self._axis_cell(f, self._up_states[c_op]))
        self._up_states[op] = self._node_cell(node.get_feature(), axis_cache)

    def _get_whole_graph_feature(self, src, end, down_graph, op2node, op2index):
        for op in end:
            self._get_down_graph_feature(op, op2node, op2index)
        for op in src:
            self._get_up_graph_feature(op, down_graph, op2node, op2index)

    def _update_region_feature(self, op, down_graph, op2node, op2index):
        node = op2node[op]
        index = op2index[op]
        for p in op.input_tensors:
            p_op = p.op
            p_node = op2node[p_op]
            p_index = op2index[p_op]
            # op must be compute operation
            axis_count = len(op.axis)
            features = node.get_axis_features()
            axis_cache = []
            for var_name, feature in features.items():
                if p_index in feature:
                    for f in feature[p_index]:
                        if f[1] >= axis_count:
                            axis_cache.append(self._reduce_axis_cell(f, self._down_states[p_op]))
                        else:
                            axis_cache.append(self._axis_cell(f, self._down_states[p_op]))
            self._down_states[op] = self._node_cell(node.get_feature(), axis_cache)
        if op in down_graph:
            for c_op in down_graph[op]:
                c_node = op2node[c_op]
                # c_op must be compute operation
                axis_count = len(c_op.axis)
                features = c_node.get_axis_features()
                axis_cache = []
                for var_name, feature in features.items():
                    if index in feature:
                        for f in feature[index]:
                            if f[1] >= axis_count:
                                axis_cache.append(self._reduce_axis_cell(f, self._up_states[c_op]))
                            else:
                                axis_cache.append(self._axis_cell(f, self._up_states[c_op]))
            self._up_states[op] = self._node_cell(node.get_feature(), axis_cache)

    def _pack_axis_feature(self, op, features, op2node, index2op):
        node = op2node[op]
        node_feature_cache = []
        for op_index, fs in features.items():
            p_node = op2node[index2op[op_index]]
            for f in fs:
                tmp = torch.cat([f, p_node.get_feature()])
                node_feature_cache.append(self._cell_1(tmp))
        return self._cell_2(node.get_feature(), node_feature_cache)

    def _pack_node_axis_feature(self, op, features, op2node, index2op):
        node = op2node[op]
        axis_cache = []
        for feature in features:
            node_feature_cache = []
            for op_index, fs in feature.items():
                p_node = op2node[index2op[op_index]]
                for f in fs:
                    tmp = torch.cat([f, p_node.get_feature()])
                    node_feature_cache.append(self._cell_1(tmp))
            axis_cache.append(self._cell_2(node.get_feature(), node_feature_cache))
        return self._cell_2(node.get_feature(), axis_cache)

    def get_parameter_size(self):
        p_size = 0
        for p in self.parameters():
            mul = 1
            for v in p.shape:
                mul *= v
            p_size += mul
        return "Total size of parameters: {}MB".format(p_size * 4 / (1024 ** 2))





