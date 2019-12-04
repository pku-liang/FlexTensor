from __future__ import absolute_import


import tvm
from auto_schedule.nn import conv2d_nchw
from expr_visitor import ExprVisitor
from graph import TensorNode, IndexNode


class ReadRelation(object):
    def __init__(self, tensor, pos):
        self.tensor = tensor
        self.pos = pos


class RelationUnion(object):
    def __init__(self):
        self.read_relation = []
        self.index_relation = []
    
    def add_read(self, tensor, pos):
        self.read_relation.append(ReadRelation(tensor, pos))


class GetRelationOp(ExprVisitor):
    def __init__(self, op):
        super(GetRelationOp, self).__init__()
        if not isinstance(op, tvm.tensor.ComputeOp):
            raise RuntimeError("Can't use for type: %s" % type(op))
        self.root_op = op
        self.call_op = None
        self.call_pos = -1
        self.call_value_index = -1
        self.var_map = {}
        for it in op.axis:
            self.var_map[it.var] = RelationUnion()
        for it in op.reduce_axis:
            self.var_map[it.var] = RelationUnion()

        self.sym_stack = []
        self.op_num_stack = []

    def visit_call(self, expr):
        self.call_op = expr.func
        self.call_value_index = expr.value_index
        for i, arg in enumerate(expr.args):
            self.call_pos = i
            self.visit(arg)
        self.call_op = None
        self.call_pos = -1

    def visit_var(self, expr):
        assert expr in self.var_map
        if self.call_pos >= 0 and self.call_op is not None and self.call_value_index >= 0:
            self.var_map[expr].add_read(self.call_op.output(self.call_value_index), self.call_pos)

    def visit_add(self, expr):
        self.sym_stack.append("+")
        self.op_num_stack.append(2)
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_sub(self, expr):
        self.sym_stack.append("-")
        self.op_num_stack.append(2)
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_mul(self, expr):
        self.sym_stack.append("*")
        self.op_num_stack.append(2)
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_div(self, expr):
        self.sym_stack.append("/")
        self.op_num_stack.append(2)
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_mod(self, expr):
        self.sym_stack.append("%")
        self.op_num_stack.append(2)
        self.visit(expr.a)
        self.visit(expr.b)
    
    def get_relation(self):
        for b in self.root_op.body:
            self.visit(b)


if __name__ == "__main__":
    N = 16
    C = 256
    H = 14
    W = 14
    K = 512
    k = 3
    Img = tvm.placeholder([N, C, H, W], dtype="float32")
    Kernel = tvm.placeholder([K, C, k, k], dtype="float32")
    Outputs = conv2d_nchw(Img, Kernel, None, 1, 1, 1, 1)

    visitor = GetRelationOp(Outputs.op)
    visitor.get_relation()

    s = tvm.create_schedule(Outputs.op)
    stmt = tvm.lower(s, [Img, Kernel, Outputs])
    