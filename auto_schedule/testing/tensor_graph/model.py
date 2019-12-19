import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax


def _softmax(src, index, num_nodes):
    out = src
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out


class MyConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_type=1, num_edge_type=1, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(MyConv, self).__init__(aggr='add', **kwargs)
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
            zeros(getattr(self, "node_bias_%d" % i))
        for i in range(self.num_edge_type):
            glorot(getattr(self, "edge_weight_%d" % i))


    def forward(self, x, node_type_index, edge_index, edge_type_index, size=None):
        assert len(node_type_index) == self.num_node_type + 1, "a=%d, b=%d" % (node_type_index, self.num_node_type)
        res = []
        for i in range(self.num_node_type):
            tmp_res = torch.matmul(x[node_type_index[i]:node_type_index[i+1]], getattr(self, "node_weight_%d" % i))
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


class ComputeGraph(object):
    def __init__(self, x, edge_index, node_type_index, edge_type_index):
        self.x = x
        self.edge_index = edge_index
        self.node_type_index = node_type_index
        self.edge_type_index = edge_type_index
        self.num_nodes = len(x)
        self.num_edges = len(edge_index[0])
        self.num_node_type = len(node_type_index) - 1
        self.num_edge_type = len(edge_type_index) - 1
        self.in_channel = len(x[0])


if __name__ == "__main__":
    N = 3
    C = 2
    K = 10
    Nnode = 3
    Nedge = 2
    net = MyConv(C, K, Nnode, Nedge)
    x = torch.rand([N, C])
    edge_index = torch.LongTensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    node_type_index = [0, 1, 2, 3]
    edge_type_index = [0, 2, 6]
    outputs = net(x, node_type_index, edge_index, edge_type_index)
    print(outputs)
