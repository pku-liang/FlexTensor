from __future__ import absolute_import

import queue
import tvm
from expr_visitor import ExprVisitor
from utils import strict_limit


class ExprNode(object):
    def __init__(self):
        self.parent = None
        self.priority = 2

    def simplify(self):
        return self


class Var(ExprNode):
    def __init__(self, var):
        super(Var, self).__init__()
        self.var = var
        self.priority = 1

    def __eq__(self, b):
        if not isinstance(b, self.__class__):
            return False
        tmp = set()
        tmp.add(self.var)
        return b.var in tmp

    def __str__(self):
        return str(self.var) + "\n"


class Val(ExprNode):
    def __init__(self, val):
        super(Val, self).__init__()
        self.val = val
        self.priority = 0

    def __eq__(self, b):
        return isinstance(b, self.__class__) and self.val == b.val

    def __str__(self):
        return str(self.val) + "\n"


class Add(ExprNode):
    def __init__(self, left, right):
        super(Add, self).__init__()
        self.left = left
        self.right = right

    def simplify(self):
        left = self.left.simplify()
        right = self.right.simplify()
        if left.priority < right.priority:
            tmp = left
            left = right
            right = tmp
        # const
        if isinstance(left, Val) and isinstance(right, Val):
            return Val(left.value + right.value)
        elif isinstance(left, Var) and isinstance(right, Var):
            if left == right:
                return Mul(left, Val(2))
        

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right
    
    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Add\n-left=%s-right=%s" % (left, right)
        return ret


class Sub(ExprNode):
    def __init__(self, left, right):
        super(Sub, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right
    
    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Sub\n-left=%s-right=%s" % (left, right)
        return ret


class Mul(ExprNode):
    def __init__(self, left, right):
        super(Mul, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Mul\n-left=%s-right=%s" % (left, right)
        return ret


class Div(ExprNode):
    def __init__(self, left, right):
        super(Div, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Div\n-left=%s-right=%s" % (left, right)
        return ret


class Mod(ExprNode):
    def __init__(self, left, right):
        super(Mod, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Mod\n-left=%s-right=%s" % (left, right)
        return ret


class And(ExprNode):
    def __init__(self, left, right):
        super(And, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "And\n-left=%s-right=%s" % (left, right)
        return ret


class Or(ExprNode):
    def __init__(self, left, right):
        super(Or, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Or\n-left=%s-right=%s" % (left, right)
        return ret


class Not(ExprNode):
    def __init__(self, left):
        super(Not, self).__init__()
        self.left = left

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left

    def __str__(self):
        left = str(self.left)
        ret = "Not\n-left=%s" % (left)
        return ret


class Min(ExprNode):
    def __init__(self, left, right):
        super(Min, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Min\n-left=%s-right=%s" % (left, right)
        return ret


class Max(ExprNode):
    def __init__(self, left, right):
        super(Max, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Max\n-left=%s-right=%s" % (left, right)
        return ret


class Eq(ExprNode):
    def __init__(self, left, right):
        super(Eq, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Eq\n-left=%s-right=%s" % (left, right)
        return ret


class Ne(ExprNode):
    def __init__(self, left, right):
        super(Ne, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Ne\n-left=%s-right=%s" % (left, right)
        return ret


class Lt(ExprNode):
    def __init__(self, left, right):
        super(Lt, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Lt\n-left=%s-right=%s" % (left, right)
        return ret


class Le(ExprNode):
    def __init__(self, left, right):
        super(Le, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Le\n-left=%s-right=%s" % (left, right)
        return ret


class Gt(ExprNode):
    def __init__(self, left, right):
        super(Gt, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Gt\n-left=%s-right=%s" % (left, right)
        return ret


class Ge(ExprNode):
    def __init__(self, left, right):
        super(Ge, self).__init__()
        self.left = left
        self.right = right

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.left == b.left and self.right == b.right

    def __str__(self):
        left = str(self.left)
        right = str(self.right)
        ret = "Ge\n-left=%s-right=%s" % (left, right)
        return ret


class Cast(ExprNode):
    def __init__(self, value):
        super(Cast, self).__init__()
        self.value = value

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return self.value == b.value

    def __str__(self):
        value = str(self.value)
        ret = "cast\n-value=%s" % (value)
        return ret


class Call(ExprNode):
    def __init__(self, children):
        super(Call, self).__init__()
        self.children = children

    def __eq__(self, b):
        if not isinstance(b, self,__class__):
            return False
        return all([self.children[i] == b.children[i] for i in range(len(self.children))])

    def __str__(self):
        ret = "call\n-args=%s" % (",".join([str(x) for x in self.children]))
        return ret


class CopyVisitor(ExprVisitor):
    def __init__(self):
        super(CopyVisitor, self).__init__()
        self.tree = None
        self.p = None

    def visit_call(self, expr):
        """
        ['args', 'call_type', 'dtype', 'func', 'name', 'value_index']
        """
        tmp = []
        for arg in expr.args:
            ret = self.visit(arg)
            tmp.append(ret)
        return Call(tmp)

    def visit_var(self, expr):
        """
        ['dtype', 'name']
        """
        return Var(expr)

    def visit_float_imm(self, expr):
        """
        ['dtype', 'value']
        """
        return Val(expr.value)

    def visit_int_imm(self, expr):
        """
        ['dtype', 'value']
        """
        return Val(expr.value)

    def visit_uint_imm(self, expr):
        """
        ['dtype', 'value']
        """
        return Val(expr.value)

    def visit_string_imm(self, expr):
        """
        ['dtype', 'value']
        """
        return Val(expr.value)

    def visit_add(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Add(left, right)

    def visit_sub(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Sub(left, right)

    def visit_div(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Div(left, right)

    def visit_mul(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Mul(left, right)

    def visit_mod(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Mod(left, right)
    
    def visit_and(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return And(left, right)

    def visit_or(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Or(left, right)

    def visit_not(self, expr):
        """
        ['a', 'dtype']
        """
        left = self.visit(expr.a)
        return Not(left)

    def visit_min(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Min(left, right)

    def visit_max(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Max(left, right)

    def visit_eq(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Eq(left, right)

    def visit_ne(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Ne(left, right)

    def visit_le(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Le(left, right)
    
    def visit_lt(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Lt(left, right)

    def visit_ge(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Ge(left, right)

    def visit_gt(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        left = self.visit(expr.a)
        right = self.visit(expr.b)
        return Gt(left, right)

    def visit_cast(self, expr):
        """
        ['value', 'dtype']
        """
        left = self.visit(expr.value)
        return Cast(left)



class IndexTree(object):
    def __init__(self):
        self.root = Val(0)
        self.tail = self.root
    
    def from_expr(self, expr):
        pass

    def add_left_node(self, parent, child):
        parent.left = child
        child.parent = parent

    def add_right_node(self, parent, child):
        parent.right = child
        child.parent = parent


def relation_from_expr(expr):
    """
    get relation from a tvm expression
    """
    relation_lst = []

    return relation_lst


def relation_from_op(op):
    """
    get relation from one op
    """
    assert isinstance(op, tvm.tensor.ComputeOp), strict_limit("tvm.tensor.ComputeOp")
    relation_lst = []
    for body in op.body:
        part_lst = relation_from_expr(body)
        relation_lst.append(part_lst)
    return relation_lst


def build_graph_from_relations(relations):
    """
    build a graph from relation list
    """
    return {}


def build(ops):
    """
    build the graph from list of op
    return a graph
    """
    # first get list of relation from ops
    relation_lst = []
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    q = queue.Queue()
    for op in ops:
        q.put(op)
    while not q.empty():
        cur_op = q.get()
        part_relation_lst = relation_from_op(cur_op)
        relation_lst.extend(part_relation_lst)
        for t in cur_op.input_tensors:
            if isinstance(t.op, tvm.tensor.ComputeOp):
                q.put(t.op)

    # then build a graph from relations
    graph = build_graph_from_relations(relation_lst)
    return graph


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

    visitor = CopyVisitor()
    res = visitor.visit(b.op.body[0])
    print(str(res))