from __future__ import absolute_import


import tvm


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

    def __str__(self):
        return str(self.data)


class IndexNode(GraphNode):
    def __init__(self, tvm_op, axis_pos, axis_type="spatial"):
        assert axis_type in ["spatial", "reduce"]
        self.data0 = tvm_op
        self.data1 = axis_pos
        self.data2 = axis_type

    def _var(self):
        if self.data2 == "spatial":
            return self.data0.axis[self.data1].var
        else:
            return self.data0.reduce_axis[self.data1].var

    def __hash__(self):
        return self._var().__hash__()

    def __eq__(self, b):
        tmp = set()
        tmp.add(self._var())
        if isinstance(b, self.__class__):
            return b._var() in tmp
        elif isinstance(b, tvm.expr.Var):
            return b in tmp
        else:
            return False

    def __str__(self):
        return str(self._var())


class ConstNode(GraphNode):
    def __init__(self, value):
        self.data = value

    def __hash__(self):
        return self.data.__hash__()

    def __eq__(self, b):
        return self.data == b

    def __str__(self):
        return str(self.data)

    
        