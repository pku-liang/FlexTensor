import tvm
import math
import time
import torch
torch.set_printoptions(threshold=100000)
import torch_geometric

import model

from flextensor.scheduler import schedule_with_config_ops
from flextensor.measure import _evaluate
from flextensor.utils import any_factor_split, Config

MAX_EXTENT = 1024


def schedule_ops(ops, bufs, config, target="llvm"):
    s, bufs = schedule_with_config_ops(ops, bufs, config, target=target)
    return s, bufs


"""[
    [
        {
            "spatial": [[32, 2, 16, 2], [8, 4, 16, 4]], 
            "reduce": [[2048, 1, 1]], 
            "inline": [], 
            "unroll": [[1, 0]]
        }
    ], 
    {
        "spatial": [], 
        "reduce": [], 
        "inline": [[0]], 
        "unroll": []
    }
   ]
"""
def gemm_config(M, N, K, logits_dict):
    spatial_split_parts = 4
    reduce_split_parts = 4
    unroll_max_factor = 10

    sy = any_factor_split(M, spatial_split_parts)
    sx = any_factor_split(N, spatial_split_parts)
    sk = any_factor_split(K, reduce_split_parts)
    unroll = []
    for i in range(1):
        for j in range(unroll_max_factor + 1):
            unroll.append([i, 2**j])

    def _rational(lst, max_val):
        return torch.FloatTensor([[y / float(max_val) for y in x] for x in lst])
    nsy = _rational(sy, M)
    nsx = _rational(sx, N)
    nsk = _rational(sk, K)
    
    n_unroll = torch.FloatTensor([[x[0] / float(2) + 0.5, math.log2(x[1]) / 1] for x in unroll])

    # get logits
    spatial_logits = logits_dict["spatial"]
    reduce_logits = logits_dict["reduce"]
    unroll_logits = logits_dict["unroll"]
    
    # make choice
    feature_size = len(logits_dict["spatial"][0])
    split_classifier = model.MLP(feature_size + spatial_split_parts)
    unroll_classifier = model.MLP(feature_size + 2)
    cy = torch.argmax(split_classifier(torch.cat([nsy, torch.zeros([nsy.shape[0], feature_size]) + spatial_logits[0]], dim=1)))
    cx = torch.argmax(split_classifier(torch.cat([nsx, torch.zeros([nsx.shape[0], feature_size]) + spatial_logits[1]], dim=1)))
    ck = torch.argmax(split_classifier(torch.cat([nsk, torch.zeros([nsk.shape[0], feature_size]) + reduce_logits[0]], dim=1)))
    cu = torch.argmax(unroll_classifier(torch.cat([n_unroll, torch.zeros([n_unroll.shape[0], feature_size]) + unroll_logits], dim=1)))

    print(cy, cx, ck, cu)
    
    # print choice
    print("Print choice")
    print("split y =", sy[cy])
    print("split x =", sx[cx])
    print("split k =", sk[ck])
    print("unroll", unroll[cu])

    # make config
    op_config = [{
        "spatial": [sy[cy], sx[cx]],
        "reduce": [sk[ck]],
        "inline": [],
        "unroll": [unroll[cu]]
    }]
    graph_config = {
        "spatial": [], 
        "reduce": [], 
        "inline": [[0]], 
        "unroll": []
    }
    return Config(op_config, graph_config)


def graph_gemm(M, N, K):
    x = torch.FloatTensor(
        [
            [M, 0],  # 0, tensor nodes
            [K, 0],  # 1
            [1, 0],  # 2
            [K, 0],  # 3
            [N, 0],  # 4
            [1, 0],  # 5
            [M, 0],  # 6
            [N, 0],  # 7
            [1, 0],  # 8
            [0, M],  # 9, spatial nodes
            [0, N],  # 10
            [0, K]   # 11, reduce nodes
            # add nodes
        ]
    ) / MAX_EXTENT
    edge_index = torch.LongTensor(
        [
            [0, 0],  # self edge
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],
            [8, 8],
            [9, 9],
            [10, 10],
            [11, 11],
            [0, 1],  # stride edge
            [1, 0],
            [1, 2],
            [2, 1],
            [3, 4],
            [4, 3],
            [4, 5],
            [5, 4],
            [6, 7],
            [7, 6],
            [7, 8],
            [8, 7],
            [0, 9],  # read edge
            [9, 0],
            [1, 10],
            [10, 1],
            [3, 10],
            [10, 3],
            [4, 11],
            [11, 4],
            [6, 9],  # write edge
            [9, 6],
            [7, 11],
            [11, 7],
            # add edge
        ]
    ).t()

    node_type_index = [0, 9, 11, 12, 12]
    edge_type_index = [0, 12, 24, 32, 36, 36]

    g = model.ComputeGraph(x, edge_index, node_type_index, edge_type_index)

    return g


if __name__ == "__main__":
    M = 512
    N = 64
    K = 128
    torch.manual_seed(100000)
    g = graph_gemm(M, N, K)
    out_channels = [32, 64, 1024, 4]
    in_channel = g.in_channel
    output = g.x
    for out_channel in out_channels:
        net = model.MyConv(in_channel, out_channel, g.num_node_type, g.num_edge_type, heads=4)
        output = net(output, g.node_type_index, g.edge_index, g.edge_type_index)
        in_channel = out_channel * 4

    print(output)
    
    unroll_output = (output[9] + output[10] + output[11]) / 3.0

    print(unroll_output)

    # show groups
    groups = list(range(g.num_nodes))
    for i in range(g.num_nodes):
        for j in range(i + 1, g.num_nodes):
            if groups[j] != j:
                continue
            if all(torch.abs(output[i] - output[j]) < 1e-5):
                groups[j] = i
    roots = set(groups)
    for id in roots:
        tmp = []
        for i in range(g.num_nodes):
            if groups[i] == id:
                tmp.append(i)
        print("Group %d: %s" % (id, str(tmp)))

    # predict
    logits = {}
    logits["spatial"] = [output[9], output[10]]
    logits["reduce"] = [output[11]]
    logits["unroll"] = unroll_output
    
    config = gemm_config(M, N, K, logits)

    # compute
    A = tvm.te.placeholder((M, K))
    B = tvm.te.placeholder((K, N))
    C = gemm(A, B)

    # schedule
    s, bufs = schedule_ops([C.op], [A, B, C], config, target="cuda")

    # build
    func = tvm.build(s, bufs, "cuda")
    print(func.imported_modules[0].get_source())

    # run
    cost = _evaluate(s, bufs, "cuda", 0, 10)
    print("time cost is:", cost)