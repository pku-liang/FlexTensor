from __future__ import absolute_import


import tvm
from auto_schedule.nn import conv2d_nchw
from expr_visitor import ExprVisitor
from graph import TensorNode, IndexNode


def gcd(a, b):
    if a > b:
        tmp = a
        a = b
        b = tmp
    while a > 0:
        r = b % a
        b = a
        a = r
    return b


def simplify(a, b):
    v = gcd(a, b)
    return (a / v, b / v)


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


class IndexInfo(object):
    def __init__(self, index):
        self.index = index
        self.add_sym = []
        self.add_const = 0
        # self.mul_sym = []
        self.mul_const = 1
        # self.div_sym = []
        self.div_const = 1
        # self.mod_sym = []
        self.mod_const = []

    def can_add_eliminate(self, b):
        for ele in self.add_sym:
            assert isinstance(ele, self.__class__)
            can = ele.can_add_eliminate(b)
            if can is not None:
                return can
        tmp = set()
        tmp.add(self.index)
        if b.index in tmp:
            int_mul_a = isinstance(self.mul_const, int)
            int_div_a = isinstance(self.div_const, int)
            no_mod_a = len(self.mod_const) == 0
            int_mul_b = isinstance(b.mul_const, int)
            int_div_b = isinstance(b.div_const, int)
            no_mod_b = len(b.mod_const) == 0
            no_add_sym_b = len(b.add_sym) == 0
            if int_mul_a and int_div_a and no_mod_a and int_mul_b and int_div_b and no_mod_b and no_add_sym_b:
                return self
        return None

    def add_eliminate(self, b):
        new_mul = self.mul_const * b.div_const + self.div_const * b.mul_const
        new_div = self.div_const * b.div_const
        new_mul, new_div = simplify(new_mul, new_div)
        self.mul_const = new_mul
        self.div_const = new_div
        self.add_const += b.add_const
        return self

    def mul(self, b):
        if isinstance(b, (int,)):
            self.mul_const *= b
            self.add_const *= b
            for sym in self.add_sym:
                if isinstance(sym, self.__class__):
                    sym.mul(b)
        else:
            raise RuntimeError("Currently no support for %s multiplication" % type(b))
        return self

    def div(self, b):   # TODO bad implementation
        if isinstance(b, (int,)):
            self.div_const *= b
            self.add_const /= b
            for sym in self.add_sym:
                if isinstance(sym, self.__class__):
                    sym.div(b)
        else:
            raise RuntimeError("Currently no support for %s division" % type(b))
        return self

    def add(self, b):
        if isinstance(b, (int,)):
            self.add_const += b
        elif isinstance(b, self.__class__):
            if self.can_add_eliminate(b):
                return self.add_eliminate(b)
            else:
                self.add_sym.append(b)
        else:
            raise RuntimeError("Currently no support for type %s addition" % type(b))
        return self

    def sub(self, b):
        if isinstance(b, (int,)):
            self.add_const -= b
        elif isinstance(b, self.__class__):
            new_b = b.mul(-1)
            self.add(new_b)
        else:
            raise RuntimeError("Currently no support for type %s substraction" % type(b))
        return self

    def mod(self, b):
        if isinstance(b, (int,)):
            self.mod_const.append(b)
        else:
            raise RuntimeError("Currently no support for variable modular")
        return self

    def __str__(self):
        return "%s * %s / %s%s" % (str(self.index), str(self.mul_const), str(self.div_const), " + ".join([""] + [str(x) for x in self.add_sym]))
            


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

        self.bak_sym_stack = []
        self.bak_op_stack = []
        self.bak_sym_count = []

        self.sym_stack = []
        self.op_num_stack = []
        self.sym_count_stack = []

    def visit_call(self, expr):
        self.call_op = expr.func
        self.call_value_index = expr.value_index
            
        for i, arg in enumerate(expr.args):
            self.bak_stack()
            self.call_pos = i
            self.visit(arg)
            print(self.sym_stack, [str(x) for x in self.sym_stack])
            self.recover_stack()
            
        self.call_op = None
        self.call_pos = -1

    def visit_var(self, expr):
        assert expr in self.var_map
        if self.call_pos >= 0 and self.call_op is not None and self.call_value_index >= 0:
            self.var_map[expr].add_read(self.call_op.output(self.call_value_index), self.call_pos)
        if len(self.sym_stack) == 0:
            return
        self.sym_stack.append(IndexInfo(expr))
        self.sym_count += 1
        print([str(x) for x in self.sym_stack])
        while len(self.op_num_stack) and self.sym_count >= self.op_num_stack[-1]:
            b = self.sym_stack.pop()    # IndexInfo
            a = self.sym_stack.pop()
            sym = self.sym_stack.pop()
            self.op_num_stack.pop()
            if isinstance(a, IndexInfo):
                if sym == "*":
                    a.mul(b)
                elif sym == "/":
                    a.div(b)
                elif sym == "+":
                    a.add(b)
                elif sym == "-":
                    a.sub(b)
                elif sym == "%":
                    a.mod(b)
                else:
                    raise RuntimeError("Unsupported operation %s" % sym)
                next_sym = a
            elif isinstance(a, (tvm.expr.IntImm, tvm.expr.UIntImm)):
                if sym == "*":
                    b.mul(a.value)
                elif sym == "/":
                    b.div(a.value)
                elif sym == "+":
                    b.add(a.value)
                elif sym == "-":
                    b.sub(a.value)
                elif sym == "%":
                    b.mod(a.vallue)
                else:
                    raise RuntimeError("Unsupported operaion %s" % sym)
                next_sym = b
            else:
                raise RuntimeError("Unsupported type %s" % type(a))
            self.sym_stack.append(next_sym)
            self.sym_count -= 1
        
    def visit_add(self, expr):
        if self.call_op is not None:
            self.sym_stack.append("+")
            self.op_num_stack.append(2)
            self.sym_count_stack.append(0)
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_sub(self, expr):
        if self.call_op is not None:
            self.sym_stack.append("-")
            self.op_num_stack.append(2)
            self.sym_count_stack.append(0)
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_mul(self, expr):
        if self.call_op is not None:
            self.sym_stack.append("*")
            self.op_num_stack.append(2)
            self.sym_count_stack.append(0)
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_div(self, expr):
        if self.call_op is not None:
            self.sym_stack.append("/")
            self.op_num_stack.append(2)
            self.sym_count_stack.append(0)
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_mod(self, expr):
        if self.call_op is not None:
            self.sym_stack.append("%")
            self.op_num_stack.append(2)
            self.sym_count_stack.append(0)
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_int_imm(self, expr):
        if self.call_op is not None:
            self.sym_stack.append(expr)
            if len(self.sym_count_stack):
                self.sym_count_stack[-1] += 1
        print([str(x) for x in self.sym_stack])
        while len(self.op_num_stack) and self.sym_count_stack[-1] == self.op_num_stack[-1]:
            self.op_num_stack.pop()
            self.sym_count_stack.pop()
            b = self.sym_stack.pop()    # int_imm
            a = self.sym_stack.pop()
            sym = self.sym_stack.pop()
            if isinstance(a, IndexInfo):
                if sym == "*":
                    a.mul(b.value)
                elif sym == "/":
                    a.div(b.value)
                elif sym == "+":
                    a.add(b.value)
                elif sym == "-":
                    a.sub(b.value)
                elif sym == "%":
                    a.mod(b.value)
                else:
                    raise RuntimeError("Unsupported operation %s" % sym)
            elif isinstance(a, int):
                if sym == "*":
                    a *= b.value
                elif sym == "/":
                    a /= b.value
                elif sym == "+":
                    a += b.value
                elif sym == "-":
                    a -= b.value
                elif sym == "%":
                    a %= b.value
                else:
                    raise RuntimeError("Unsupported operaion %s" % sym)
            else:
                raise RuntimeError("Unsupported type %s" % type(a))
            self.sym_stack.append(a)
        
    def visit_uint_imm(self, expr):
        self.visit_int_imm(expr)

    def get_relation(self):
        for b in self.root_op.body:
            print(b)
            self.visit(b)
            print(len(self.sym_stack))
            print(len(self.op_num_stack))
            print(self.sym_count)

    def bak_stack(self):
        self.bak_sym_stack.append(self.sym_stack)
        self.bak_op_stack.append(self.op_num_stack)
        self.bak_sym_count.append(self.sym_count)
        self.sym_stack.clear()
        self.op_num_stack.clear()
        self.sym_count = 0

    def recover_stack(self):
        self.sym_stack = self.bak_sym_stack.pop()
        self.op_num_stack = self.bak_op_stack.pop()
        self.sym_count = self.bak_sym_count.pop()


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

    a = tvm.placeholder([4, 4])
    b = tvm.compute([4, 4], lambda i, j: a[(3 * i + j) * 4 - 12 * i - 4 * j + i, j])

    visitor = GetRelationOp(b.op)
    visitor.get_relation()

    s = tvm.create_schedule(Outputs.op)
    stmt = tvm.lower(s, [Img, Kernel, Outputs])
    