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

    def __str__(self):
        return "[tensor=%s, pos=%s]" % (self.tensor, self.pos)


class WriteRelation(object):
    def __init__(self, tensor, pos):
        self.tensor = tensor
        self.pos = pos

    def __str__(self):
        return "[tensor=%s, pos=%s]" % (self.tensor, self.pos)


class IndexRelation(object):
    def __init__(self, index, rel_type="add"):
        self.index = index
        self.rel_type = rel_type

    def __str__(self):
        return "[index=%s, type=%s]" % (self.index, self.rel_type)


class ConstRelation(object):
    def __init__(self, const, rel_type):
        self.const = const
        self.rel_type = rel_type

    def __str__(self):
        return "[const=%s, type=%s]" % (self.const, self.rel_type)


class RelationUnion(object):
    def __init__(self):
        self.read_relation = []
        self.index_relation = []
        self.const_relation = []
    
    def add_read(self, tensor, pos):
        self.read_relation.append(ReadRelation(tensor, pos))

    def add_index(self, index, rel_type="add"):
        self.index_relation.append(IndexRelation(index, rel_type=rel_type))

    def add_const(self, const, rel_type):
        self.const_relation.append(ConstRelation(const, rel_type))

    def __str__(self):
        string = "RelationUnion(\n"
        string += "    read=[%s]\n" % ", ".join([str(x) for x in self.read_relation])
        string += "    index=[%s]\n" % ", ".join([str(x) for x in self.index_relation])
        string += "    const=[%s]\n" % ", ".join([str(x) for x in self.const_relation])
        string += ")\n"
        return string


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

    def simplify(self):
        lst = self._simplify()
        return lst[0]

    def _simplify(self):
        left_const = 0
        res_lst = []
        new_mul, new_div = simplify(self.mul_const, self.div_const)
        self.mul_const = new_mul
        self.div_const = new_div
        if abs(self.mul_const) >= 1e-5:
            return [self]
        elif abs(self.add_const) >= 1e-5:
            left_const = self.add_const
        for sym in self.add_sym:
            res = sym.simplify()
            if isinstance(res, self.__class__):
                res.add(left_const)
                res_lst.append(res)
            elif isinstance(res, int):
                left_const += res
        res_lst.append(left_const)
        return res_lst

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
        if isinstance(b, (tvm.expr.IntImm, tvm.expr.UIntImm)):
            self.mul_const *= b.value
            self.add_const *= b.value
            for sym in self.add_sym:
                if isinstance(sym, self.__class__):
                    sym.mul(b.value)
        elif isinstance(b, (int,)):
            self.mul_const *= b
            self.add_const *= b
            for sym in self.add_sym:
                if isinstance(sym, self.__class__):
                    sym.mul(b)
        else:
            raise RuntimeError("Currently no support for %s multiplication" % type(b))
        return self

    def div(self, b):   # TODO bad implementation
        if isinstance(b, (tvm.expr.IntImm, tvm.expr.UIntImm)):
            self.div_const *= b.value
            self.add_const /= b.value
            for sym in self.add_sym:
                if isinstance(sym, self.__class__):
                    sym.div(b.value)
        elif isinstance(b, (int,)):
            self.div_const *= b
            self.add_const /= b
            for sym in self.add_sym:
                if isinstance(sym, self.__class__):
                    sym.div(b)
        else:
            raise RuntimeError("Currently no support for %s division" % type(b))
        return self

    def add(self, b):
        if isinstance(b, (tvm.expr.IntImm, tvm.expr.UIntImm)):
            self.add_const += b
        if isinstance(b, (int,)):
            self.add_const += b
        elif isinstance(b, self.__class__):
            eliminate = self.can_add_eliminate(b)
            if eliminate is not None:
                eliminate.add_eliminate(b)
            else:
                self.add_sym.append(b)
        else:
            raise RuntimeError("Currently no support for type %s addition" % type(b))
        return self

    def sub(self, b):
        if isinstance(b, (tvm.expr.IntImm, tvm.expr.UIntImm)):
            self.add_const -= b
        if isinstance(b, (int,)):
            self.add_const -= b
        elif isinstance(b, self.__class__):
            new_b = b.mul(-1)
            self.add(new_b)
        else:
            raise RuntimeError("Currently no support for type %s substraction" % type(b))
        return self

    def mod(self, b):
        if isinstance(b, (tvm.expr.IntImm, tvm.expr.UIntImm)):
            self.mod_const.append(b)
        if isinstance(b, (int,)):
            self.mod_const.append(b)
        else:
            raise RuntimeError("Currently no support for variable modular")
        return self

    def __str__(self):
        return "%s * %s / %s mod [%s]%s" % (str(self.index), str(self.mul_const), 
        str(self.div_const), ",".join([str(x) for x in self.mod_const]), " + ".join([""] + [str(x) for x in self.add_sym]))
            


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

        self.res_lst = []

    def visit_call(self, expr):
        self.call_op = expr.func
        self.call_value_index = expr.value_index
            
        for i, arg in enumerate(expr.args):
            self.bak_stack()
            self.call_pos = i
            self.visit(arg)
            if len(self.sym_stack):
                res = self.sym_stack[0]
                if isinstance(res, IndexInfo):
                    res = res.simplify()
                self.res_lst.append(res)
            self.recover_stack()
            
        self.call_op = None
        self.call_pos = -1

    def visit_var(self, expr):
        assert expr in self.var_map
        if self.call_op is not None:
            self.var_map[expr].add_read(self.call_op.output(self.call_value_index), self.call_pos)
            self.sym_stack.append(IndexInfo(expr))
            if len(self.sym_count_stack):
                self.sym_count_stack[-1] += 1
        while len(self.op_num_stack) and self.sym_count_stack[-1] == self.op_num_stack[-1]:
            b = self.sym_stack.pop()    
            a = self.sym_stack.pop()
            sym = self.sym_stack.pop()
            self.op_num_stack.pop()
            self.sym_count_stack.pop()
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
            elif isinstance(a, (int,)):
                if sym == "*":
                    b.mul(a)
                elif sym == "/":
                    b.div(a)
                elif sym == "+":
                    b.add(a)
                elif sym == "-":
                    b.sub(a)
                elif sym == "%":
                    b.mod(a)
                else:
                    raise RuntimeError("Unsupported operaion %s" % sym)
                next_sym = b
            else:
                raise RuntimeError("Unsupported type %s" % type(a))
            self.sym_stack.append(next_sym)
            if len(self.sym_count_stack):
                self.sym_count_stack[-1] += 1
        
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
            self.sym_stack.append(expr.value)
            if len(self.sym_count_stack):
                self.sym_count_stack[-1] += 1
        while len(self.op_num_stack) and self.sym_count_stack[-1] == self.op_num_stack[-1]:
            self.op_num_stack.pop()
            self.sym_count_stack.pop()
            b = self.sym_stack.pop()    # int_imm
            a = self.sym_stack.pop()
            sym = self.sym_stack.pop()
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
                res = a
            elif isinstance(a, (int,)) and isinstance(b, (int,)):
                if sym == "*":
                    a *= b
                elif sym == "/":
                    a /= b
                elif sym == "+":
                    a += b
                elif sym == "-":
                    a -= b
                elif sym == "%":
                    a %= b
                else:
                    raise RuntimeError("Unsupported operaion %s" % sym)
                res = a
            elif isinstance(a, (int,)) and isinstance(b, (IndexInfo,)):
                if sym == "*":
                    b.mul(a)
                elif sym == "/":
                    b.div(a)
                elif sym == "+":
                    b.add(a)
                elif sym == "-":
                    b.sub(a)
                elif sym == "%":
                    b.mod(a)
                else:
                    raise RuntimeError("Unsupported operaion %s" % sym)
                res = b
            else:
                raise RuntimeError("Unsupported type a=%s, b=%s" % (type(a), type(b)))
            self.sym_stack.append(res)
            if len(self.sym_count_stack):
                self.sym_count_stack[-1] += 1
        
    def visit_uint_imm(self, expr):
        self.visit_int_imm(expr)

    def _get_relation(self, res):
        if res.index in self.var_map:
            self.var_map[res.index].add_const(res.add_const, "add")
            self.var_map[res.index].add_const(res.mul_const, "mul")
            self.var_map[res.index].add_const(res.div_const, "div")
            for v in res.mod_const:
                self.var_map[res.index].add_const(v, "mod")
            for ele in res.add_sym:
                self.var_map[res.index].add_index(ele.index, "add")
                self._get_relation(ele)

    def get_relation(self):
        for b in self.root_op.body:
            self.visit(b)
        for res in self.res_lst:
            if isinstance(res, IndexInfo):
                self._get_relation(res)
        return self.var_map
     
    def bak_stack(self):
        self.bak_sym_stack.append(self.sym_stack)
        self.bak_op_stack.append(self.op_num_stack)
        self.bak_sym_count.append(self.sym_count_stack)
        self.sym_stack.clear()
        self.op_num_stack.clear()
        self.sym_count = 0

    def recover_stack(self):
        self.sym_stack = self.bak_sym_stack.pop()
        self.op_num_stack = self.bak_op_stack.pop()
        self.sym_count_stack = self.bak_sym_count.pop()

    