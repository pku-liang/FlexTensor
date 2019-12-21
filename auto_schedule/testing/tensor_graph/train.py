import tvm
import json
import utils
import graph
import model
import space
import ops
import numpy as np
from collections import namedtuple, deque


TargetItem = namedtuple("TargetItem", "target dev_id")
DataItem = namedtuple("DataItem", "op shape target config")


def load_data(filename):
    ret = []
    with open(filename, "r") as fin:
        for line in fin:
            if line:
                key, string = line.split(":", 1)
                op, _, shape_str, target_str = key.split("_")
                shape = [int(x) for x in shape_str[1:-1].split(", ")]
                target, dev_id_str = target_str[:-1].split("(")
                dev_id = int(dev_id_str)
                config = json.loads(string)
                ret.append(DataItem(
                    op=op, 
                    shape=shape, 
                    target=TargetItem(target=target, dev_id=dev_id), 
                    config=utils.Config(config[0], config[1]))
                )
    return ret


def split_train_test_data(data, train_ratio, shuffle=True):
    num_train = int(len(data) * train_ratio)
    if shuffle:
        np.random.shuffle(data)
    train_data = data[:num_train]
    test_data = data[num_train:]
    return train_data, test_data


def flatten_graph(ops):
    bfs_order = []
    down_graph = {}
    visited = set()
    q = deque()
    for op in ops:
        q.append(op)
        visited.add(op)
    while q:
        cur = q.popleft()
        if isinstance(cur, tvm.tensor.ComputeOp):
            bfs_order.append(cur)
        for t in cur.input_tensors:
            if t.op not in visited:
                visited.add(t.op)
                q.append(t.op)
            if t not in down_graph:
                down_graph[t] = []
            down_graph[t].append(cur)
    return list(reversed(bfs_order)), down_graph


def get_compute(op, shape):
    if op == "gemm":
        return ops.GEMM(*shape)
    elif op == "conv2d":
        return ops.Conv2d(*shape)
    else:
        raise RuntimeError("Not supported op compute type: %s" % op)


def get_graph(op, shape):
    if op == "gemm":
        return graph.graph_gemm(*shape)
    else:
        raise RuntimeError("Not supported op graph type: %s" % op)


def get_space(tvm_op):
    _, down_graph = flatten_graph([tvm_op])
    return space.generate_space_intra_op(tvm_op, down_graph)
        


def train():
    raw_data = load_data("data.txt")
    train_data, test_data = split_train_test_data(raw_data, 0.8)
    
    epoch = 10

    net = model.GNNScheduler(4, num_node_type=4, num_edge_type=5)

    for ep in range(epoch):
        for data in train_data:
            graph = get_graph(data.op, data.shape)
            tvm_op_lst, tvm_bufs = get_compute(data.op, data.shape)
            tvm_op = tvm_op_lst[0]
            schedule_space = get_space(tvm_op)
            schedule_choices = {}
            for key, subspace in schedule_space.subspaces.items():
                print(key)
                schedule_choices[key] = subspace.static_entities
            logits_dict = net(graph.x, graph.node_type_index, graph.edge_index, graph.edge_type_index, schedule_choices)



if __name__ == "__main__":
    train()