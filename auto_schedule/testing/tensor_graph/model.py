import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.utils import softmax
from softmax_issue import softmax2 as softmax


class TensorGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_type=1, num_edge_type=1, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(TensorGAT, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type

        for i in range(self.num_node_type):
            setattr(self, "node_weight_%d" % i, Parameter(torch.Tensor(in_channels, heads * out_channels)))
        
        for i in range(self.num_edge_type):
            setattr(self, "edge_weight_%d" % i, Parameter(torch.Tensor(1, heads, 2 * out_channels)))

        if bias and concat:
            for i in range(self.num_node_type):
                setattr(self, "node_bias_%d" % i, Parameter(torch.Tensor(heads * out_channels)))
        elif bias and not concat:
            for i in range(self.num_node_type):
                setattr(self, "node_bias_%d" % i, Parameter(torch.Tensor(out_channels)))
        else:
            for i in range(self.num_node_type):
                self.register_parameter('node_bias_%d' % i, None)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_node_type):
            glorot(getattr(self, "node_weight_%d" % i))
            # torch.nn.init.uniform(getattr(self, "node_weight_%d" % i))
            zeros(getattr(self, "node_bias_%d" % i))
        for i in range(self.num_edge_type):
            glorot(getattr(self, "edge_weight_%d" % i))
            # torch.nn.init.uniform(getattr(self, "edge_weight_%d" % i))


    def forward(self, x, node_type_index, edge_index, edge_type_index, size=None):
        assert len(node_type_index) == self.num_node_type + 1, "a=%d, b=%d" % (len(node_type_index), self.num_node_type + 1)
        res = []
        for i in range(self.num_node_type):
            tmp_res = torch.matmul(x[node_type_index[i]:node_type_index[i+1]], getattr(self, "node_weight_%d" % i))
            tmp_res = torch.relu(tmp_res)
            res.append(tmp_res)
        linear = torch.cat(res)

        return self.propagate(edge_index, size=size, x=linear, 
            node_type_index=node_type_index, edge_type_index=edge_type_index)


    def message(self, edge_index_i, x_i, x_j, size_i, edge_type_index):
        # Compute attention coefficients.
        assert len(edge_type_index) == self.num_edge_type + 1
        x_j = x_j.view(-1, self.heads, self.out_channels)

        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha_lst = []

        for i in range(0, self.num_edge_type):
            beg = edge_type_index[i]
            end = edge_type_index[i + 1]
            alpha_tmp = (torch.cat([x_i[beg:end], x_j[beg:end]], dim=-1) * getattr(self, "edge_weight_%d" % i)).sum(dim=-1)
            alpha_lst.append(alpha_tmp)

        alpha = torch.cat(alpha_lst)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out, node_type_index):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if hasattr(self, "node_bias_0"):
            for i in range(self.num_node_type):
                beg = node_type_index[i]
                end = node_type_index[i + 1]
                aggr_out[beg:end] = aggr_out[beg:end] + getattr(self, "node_bias_%d" % i)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class MLP(nn.Module):
    def __init__(self, left_channel, right_channel, out_channel):
        super(MLP, self).__init__()
        self.left = nn.Linear(left_channel, 128)
        self.right = nn.Linear(right_channel, 128)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, out_channel)
    
    def forward(self, left, right):
        left_res = self.left(left)
        right_res = self.right(right)
        tmp = torch.cat([left_res, right_res], dim=1)
        tmp = torch.relu(tmp)
        tmp = torch.relu(self.l1(tmp))
        tmp = self.l2(tmp)
        return tmp


class GNNScheduler(nn.Module):
    def __init__(
        self, in_channel, spatial_channel=4, reduce_channel=3, 
        fuse_channel=3, reorder_channel=1, unroll_channel=2, 
        num_node_type=1, num_edge_type=1):
        super(GNNScheduler, self).__init__()
        self.spatial_channel = spatial_channel
        self.reduce_channel = reduce_channel
        self.fuse_channel = fuse_channel
        self.reorder_channel = reorder_channel
        self.unroll_channel = unroll_channel

        self.layer1 = TensorGAT(in_channel, 64, num_node_type=num_node_type, num_edge_type=num_edge_type, heads=1)
        self.layer2 = TensorGAT(64, 64, num_node_type=num_node_type, num_edge_type=num_edge_type, heads=1)
        self.layer3 = TensorGAT(32 * 2, 128, num_node_type=num_node_type, num_edge_type=num_edge_type, heads=2)
        self.layer4 = TensorGAT(128 * 2, 64, num_node_type=num_node_type, num_edge_type=num_edge_type, heads=1)
        self.spatial_classifier = MLP(spatial_channel, 64, 1)
        self.reduce_classifier = MLP(reduce_channel, 64, 1)
        self.fuse_classifier = MLP(fuse_channel, 64, 1)
        self.reorder_classifier = MLP(reorder_channel, 64, 1)
        self.unroll_classifier = MLP(unroll_channel, 64, 1)

        # debug
        self.last_act4 = None

    def forward(self, x, node_type_index, edge_index, edge_type_index, schedule_choices):
        """
        args:
        -------
        x:                torch.FloatTensor [num_nodes, feature_size]
        node_type_index:  list [tensor_nodes|spatial_nodes|reduce_nodes|add_nodes]
        edge_index:       torch.LongTensor [2, num_edges]
        edge_type_index:  list [self_edge|stride_edge|read_edge|write_edge|add_edge]
        schedule_choices: dict {knob_name(spatial/reduce/fuse/reorder/unroll):torch.FloatTensor [num_choice, channel]}
        --------
        return:
        --------
        dict {knob_name(spatial/reduce/fuse/reorder/unroll):torch.FloatTensor [num_choice, 1]}
        """
        output1 = self.layer1(x, node_type_index, edge_index, edge_type_index)
        act1 = torch.relu(output1)
        output2 = self.layer2(act1, node_type_index, edge_index, edge_type_index)
        act2 = torch.relu(output2)
        output3 = self.layer3(act2, node_type_index, edge_index, edge_type_index)
        act3 = torch.relu(output3)
        output4 = self.layer4(act3, node_type_index, edge_index, edge_type_index)
        act4 = torch.relu(output4 + output2)

        # different type of nodes
        tensor_nodes_act = act4[node_type_index[0]:node_type_index[1]]
        spatial_nodes_act = act4[node_type_index[1]:node_type_index[2]]
        reduce_nodes_act = act4[node_type_index[2]:node_type_index[3]]
        add_nodes_act = act4[node_type_index[3]:node_type_index[3]]

        ret = {}
        # spatial
        ret["spatial"] = []
        for spatial_choices, act in zip(schedule_choices["spatial"], spatial_nodes_act):
            spatial_logits = self.spatial_classifier(spatial_choices, act.expand([spatial_choices.shape[0], -1]))
            spatial_logits = torch.softmax(spatial_logits, dim=0)
            ret["spatial"].append(spatial_logits)
        # reduce
        ret["reduce"] = []
        for reduce_choices, act in zip(schedule_choices["reduce"], reduce_nodes_act):
            reduce_logits = self.reduce_classifier(reduce_choices, act.expand([reduce_choices.shape[0], -1]))
            reduce_logits = torch.softmax(reduce_logits, dim=0)
            ret["reduce"].append(reduce_logits)
        # fuse
        ret["fuse"] = []
        fuse_act = torch.mean(spatial_nodes_act, dim=0)
        for fuse_choices in schedule_choices["fuse"]:
            fuse_logits = self.fuse_classifier(fuse_choices, fuse_act.expand([fuse_choices.shape[0], -1]))
            fuse_logits = torch.softmax(fuse_logits, dim=0)
            ret["fuse"].append(fuse_logits)
        # reorder
        ret["reorder"] = []
        reorder_act = torch.mean(act4[node_type_index[1]:node_type_index[3]], dim=0)
        for reorder_choices in schedule_choices["reorder"]:
            reorder_logits = self.reorder_classifier(reorder_choices, reorder_act.expand([reorder_choices.shape[0], -1]))
            reorder_logits = torch.softmax(reorder_logits, dim=0)
            ret["reorder"].append(reorder_logits)
        # unroll
        ret["unroll"] = []
        unroll_act = reorder_act
        for unroll_choices in schedule_choices["unroll"]:
            unroll_logits = self.unroll_classifier(unroll_choices, unroll_act.expand([unroll_choices.shape[0], -1]))
            unroll_logits = torch.softmax(unroll_logits, dim=0)
            ret["unroll"].append(unroll_logits)

        return ret

    def print_grad(self):
        for p in self.parameters():
            print(p.name)
            print(torch.max(p.grad), torch.min(p.grad))
