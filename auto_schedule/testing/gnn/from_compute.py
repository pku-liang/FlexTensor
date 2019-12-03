from __future__ import absolute_import

import queue
import tvm
from graph import TensorNode, IndexNode


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
        self.data_relation = {}      # tensor -> tensor
        self.edge_types = []

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
                self.write_relation[node].append((TensorNode(op.output(i)), count))

    def set_read_relation(self, op):
        pass

    def set_index_relation(self, op):
        pass

    def set_data_relation(self, graph):
        pass

    def add_data_relation(self, from_tensor, to_tensor):
        pass

    def set_all_relation(self, graph):
        pass





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
            data.set_write_relation(op)

            
    