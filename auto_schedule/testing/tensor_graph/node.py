from __future__ import absolute_import

import tvm
from utils import strict_limit


class Node(object):
    def __init__(self, feature, name=None):
        if not isinstance(feature, (list, tuple)):
            feature = [feature]
        self.feature = feature
        self.name = name


def make_nodes_from_tensor(tensor):
    """
    return: list of Node
    """
    assert isinstance(tensor, tvm.tensor.Tensor), strict_limit("tvm.tensor.Tensor")
    node_lst = []
    for dim, val in enumerate(tensor.shape):
        assert isinstance(val, tvm.expr.IntImm), strict_limit("tvm.expr.IntImm")
        node_lst.append(Node(val.value), name="%s/%d" % (tensor.name, dim))
    return node_lst


def make_node_from_var(var, feature):
    assert isinstance(var, tvm.expr.Var), strict_limit("tvm.expr.Var")
    return Node(feature, name=var.name)

