from __future__ import absolute_import

import queue
import tvm
from graph import TensorNode, IndexNode
from get_relation import GetRelationOp, WriteRelation


def var_name(op, var):
    return str(op) + "/" + var.name


def traverse_body(body, read_relation, index_relation):
    pass


class Data(object):
    def __init__(self):
        self.tensor_map = {}     # map to tensor feature position
        self.tensor_node_features = [1]      # default special node value 1
        self.count_tensor_node = 1
        self.index_map = {}      # map to index feature position
        self.index_node_fetures = []
        self.count_index_node = 0
        self.write_relation = {}     # index -> [(tensor, pos)]
        self.read_relation = {}      # index -> [(tensor, pos)]
        self.index_relation = {}     # index -> [(index, type)]
        self.const_relation = {}     # index -> [(const, type)]

    def has_tensor(self, node):
        return node in self.tensor_map

    def add_tensor_nodes(self, tensor):
        node = TensorNode(tensor)
        assert not self.has_tensor(node)
        self.tensor_map[node] = []
        for v in tensor.shape:
            self.tensor_node_features.append(v.value)
            self.tensor_map[node].append(self.count_tensor_node)
            self.count_tensor_node += 1
        self.tensor_map[node].append(0)
    
    def has_index(self, node):
        return node in self.index_map

    def add_index_nodes(self, op):
        for count, it in enumerate(op.axis):
            node = IndexNode(op, count, "spatial")
            assert not self.has_index(node)
            self.index_map[node] = self.count_index_node
            self.index_node_fetures.append(it.dom.extent.value)
            self.count_index_node += 1

        for count, it in enumerate(op.reduce_axis):
            node = IndexNode(op, count, "reduce")
            assert not self.has_index(node)
            self.index_map[node] = self.count_index_node
            self.index_node_fetures.append(it.dom.extent.value)
            self.count_index_node += 1

    def set_write_relation(self, op):
        for count, it in enumerate(op.axis):
            node = IndexNode(op, count, "spatial")
            self.write_relation[node] = []
            for i in range(op.num_outputs):
                self.write_relation[node].append(WriteRelation(op.output(i), count))

    def set_op_relation(self, op):
        self.set_write_relation(op)
        visitor = GetRelationOp(op)
        var_map = visitor.get_relation()
        for var, relation in var_map.items():
            self.read_relation[var] = relation.read_relation
            self.index_relation[var] = relation.index_relation
            self.const_relation[var] = relation.const_relation

    def __str__(self):
        string = "Data(\n"
        tmp = []
        for node, idx in self.tensor_map.items():
            feature = [self.tensor_node_features[x] for x in idx]
            tmp.append("(tensor_node=%s, idx=%s, feature=%s)" % (node, idx, feature))
        string += "     tensor nodes: %s\n" % ", ".join(tmp)
        tmp = []
        for node, idx in self.index_map.items():
            feature = self.index_node_fetures[idx]
            tmp.append("(index_node=%s, idx=%d, feature=%s)" % (node, idx, feature))
        string += "     index nodes: %s\n" % ", ".join(tmp)
        string += "##################################\n"
        string += "read relations:\n"
        for var, relation in self.read_relation.items():
            idx = self.index_map[var]
            tmp = []
            for ele in relation:
                tmp.append("(tensor=%s, pos=%d)" % (ele.tensor, ele.pos))
            string += "    index_node (id=%d, var=%s) read from %s\n" % (idx, var, ", ".join(tmp))
        string += "##################################\n"
        string += "write relations:\n"
        for var, relation in self.write_relation.items():
            idx = self.index_map[var]
            tmp = []
            for ele in relation:
                tmp.append("(tensor=%s, pos=%d)" % (ele.tensor, ele.pos))
            string += "    index_node (id=%d, var=%s) write to %s\n" % (idx, var, ", ".join(tmp))
        string += "##################################\n"
        string += "index relations:\n"
        for var, relation in self.index_relation.items():
            idx = self.index_map[var]
            for ele in relation:
                to_idx = self.index_map[ele.index]
                string += "    index_node (id=%d, var=%s) %s index_node (id=%d, var=%s)\n" % (idx, var, ele.rel_type, to_idx, ele.index)
        string += "##################################\n"
        string += "const relations:\n"
        for var, relation in self.const_relation.items():
            idx = self.index_map[var]
            for ele in relation:
                string += "    index_node (id=%d, var=%s) %s const (val=%d)\n" % (idx, var, ele.rel_type, ele.const)
        string += "#################################\n"
        string += ")\n"
        return string


def build(ops):
    if not isinstance(ops, (list, tuple)):
        ops = [ops]

    q = queue.Queue()
    for op in ops:
        q.put(op)

    data = Data()
    visited = set()

    while not q.empty():
        op = q.get()
        if op not in visited:       # add unvisited ops
            visited.add(op)
            for t in op.input_tensors:
                q.put(t.op)
        else:
            continue
        for i in range(op.num_outputs):     # add tensor nodes
            out_tensor = op.output(i)
            if not data.has_tensor(out_tensor):
                data.add_tensor_nodes(out_tensor)
        if isinstance(op, tvm.tensor.ComputeOp):    # add index node
            data.add_index_nodes(op)
            data.set_op_relation(op)

    return data

            
if __name__ == "__main__":
    from auto_schedule.nn import conv2d_nchw
    N = 16
    C = 256
    H = 14
    W = 14
    K = 512
    k = 3
    Img = tvm.placeholder([N, C, H, W], dtype="float32")
    Kernel = tvm.placeholder([K, C, k, k], dtype="float32")
    Outputs = conv2d_nchw(Img, Kernel, None, 1, 1, 1, 1)

    a = tvm.placeholder([4, 4])
    b = tvm.compute([4, 4], lambda i, j: a[(3 * i + j) * 4 - 12 * i - 4 * j + i, j])

    data = build(Outputs.op)
    var1 = IndexNode(Outputs.op, 0, "spatial")
    var2 = Outputs.op.axis[0].var
    var3 = Outputs.op.body[0].source[0].a.args[0]
    print(var1, var2, var3)
    tmp = {}
    tmp[var1] = 999
    print(tmp[var2], tmp[var3])
    print(str(data))
