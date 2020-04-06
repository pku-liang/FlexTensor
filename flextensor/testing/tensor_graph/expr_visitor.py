from __future__ import absolute_import

import tvm


class ExprVisitor(object):
    def visit(self, expr):
        type_info = type(expr)
        if type_info == tvm.tir.Reduce:
            res = self.visit_reduce(expr)
        elif type_info == tvm.tir.Call:
            res = self.visit_call(expr)
        elif type_info == tvm.tir.Var:
            res = self.visit_var(expr)
        elif type_info == tvm.tir.FloatImm:
            res = self.visit_float_imm(expr)
        elif type_info == tvm.tir.IntImm:
            res = self.visit_int_imm(expr)
        elif type_info == tvm.tir.UIntImm:
            res = self.visit_uint_imm(expr)
        elif type_info == tvm.tir.StringImm:
            res = self.visit_string_imm(expr)
        elif type_info == tvm.tir.Add:
            res = self.visit_add(expr)
        elif type_info == tvm.tir.Sub:
            res = self.visit_sub(expr)
        elif type_info == tvm.tir.Mul:
            res = self.visit_mul(expr)
        elif type_info == tvm.tir.Div:
            res = self.visit_div(expr)
        elif type_info == tvm.tir.Mod:
            res = self.visit_mod(expr)
        elif type_info == tvm.tir.And:
            res = self.visit_and(expr)
        elif type_info == tvm.tir.Or:
            res = self.visit_or(expr)
        elif type_info == tvm.tir.Not:
            res = self.visit_not(expr)
        elif type_info == tvm.tir.Cast:
            res = self.visit_cast(expr)
        elif type_info == tvm.tir.EQ:
            res = self.visit_eq(expr)
        elif type_info == tvm.tir.NE:
            res = self.visit_ne(expr)
        elif type_info == tvm.tir.LT:
            res = self.visit_lt(expr)
        elif type_info == tvm.tir.LE:
            res = self.visit_le(expr)
        elif type_info == tvm.tir.GT:
            res = self.visit_gt(expr)
        elif type_info == tvm.tir.GE:
            res = self.visit_ge(expr)
        elif type_info == tvm.tir.Min:
            res = self.visit_min(expr)
        elif type_info == tvm.tir.Max:
            res = self.visit_max(expr)
        else:
            raise RuntimeError("Unsupported type: %s" % type_info)
        return res
        
    def visit_reduce(self, expr):
        """
        ['axis', 'combiner', 'condition', 'dtype', 'source', 'value_index']
        """
        for src in expr.source:
            self.visit(src)

    def visit_call(self, expr):
        """
        ['args', 'call_type', 'dtype', 'func', 'name', 'value_index']
        """
        for arg in expr.args:
            self.visit(arg)

    def visit_var(self, expr):
        """
        ['dtype', 'name']
        """
        pass

    def visit_float_imm(self, expr):
        """
        ['dtype', 'value']
        """
        pass

    def visit_int_imm(self, expr):
        """
        ['dtype', 'value']
        """
        pass

    def visit_uint_imm(self, expr):
        """
        ['dtype', 'value']
        """
        pass

    def visit_string_imm(self, expr):
        """
        ['dtype', 'value']
        """
        pass

    def visit_add(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_sub(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_div(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_mul(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_mod(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_and(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_or(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_not(self, expr):
        """
        ['a', 'dtype']
        """
        self.visit(expr.a)

    def visit_min(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_max(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_eq(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_ne(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_le(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)
    
    def visit_lt(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_ge(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_gt(self, expr):
        """
        ['a', 'b', 'dtype']
        """
        self.visit(expr.a)
        self.visit(expr.b)

    def visit_cast(self, expr):
        """
        ['value', 'dtype']
        """
        self.visit(expr.value)
