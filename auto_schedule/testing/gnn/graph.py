from __future__ import absolute_import


class GraphNode(object):
    def __init__(self):
        pass

    def __ne__(self, b):
        return not self.__eq__(b)


class GraphEdge(object):
    def __init__(self):
        pass

    def __ne__(self, b):
        return not self.__eq__(b)


class TensorNode(GraphEdge):
    def __init__(self, tvm_tensor):
        super(TensorNode, self).__init__()
        self.data = tvm_tensor

    def __hash__(self):
        return self.data.__hash__()

    def __eq__(self, b):
        return isinstance(b, self.__class__) and self.data == b.data


class IndexNode(GraphNode):
    def __init__(self, tvm_op, axis_pos, axis_type="spatial"):
        assert axis_type in ["spatial", "reduce"]
        self.data0 = tvm_op
        self.data1 = axis_pos
        self.data2 = axis_type

    def __hash__(self):
        tmp = 0 if self.data2 == "spatial" else 1
        return self.data0.__hash__() * 100 + self.data1 * 10 + tmp

    def __eq__(self, b):
        return (isinstance(b, self.__class__) and
                ((self.data0 == b.data0) and
                (self.data1 == b.data1) and
                (self.data2 == b.data2)))


if __name__ == "__main__":
    import tvm
    from auto_schedule.examples import FUNC_TABLE
    func = FUNC_TABLE["conv3d_channel_batch"].func
    args = FUNC_TABLE["conv3d_channel_batch"].args
    op, bufs = func(*args)
    tmp = {}
    node = TensorNode(op.output(0))
    node1 = TensorNode(op.output(0))
    node2 = TensorNode(op.input_tensors[0])
    node3 = IndexNode(op, 0, "spatial")
    node4 = IndexNode(op, 1, "spatial")
    tmp[node1] = 0
    print(node2 in tmp)
    print(node3 in tmp)
    tmp[node3] = 1
    print(node4 in tmp, node == node1)
    
        