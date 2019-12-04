from __future__ import absolute_import


import tvm
from auto_schedule.nn import conv2d_nchw
from expr_visitor import ExprVisitor
from graph import TensorNode, IndexNode


class GetRelationOp(ExprVisitor):
    def __init__(self, op):
        super(GetRelationOp, self).__init__()
        self.root_op = op
        self.call_op = None

        for b in op.body:
            self.visit(b)

    def visit_call(self, expr):
        self.call_op = expr.func
        for arg in expr.args:
            self.visit(arg)

    def visit_var(self, expr):
        for it in self.root_op.reduce_axis:
            if it.var.__hash__() == expr.__hash__():
                print(it.var == expr)
    
    def get_relation(self):
        self.visit(self.root_op.body[0])


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
    