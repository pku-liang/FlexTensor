from __future__ import absolute_import

import tvm


class ExprVisitor(object):
    def visit(self, expr):
        type_info = type(expr)
        if type_info == tvm.tir.Reduce:
            self.visit_reduce(expr)
        elif type_info == tvm.tir.Call:
            self.visit_call(expr)
        elif type_info == tvm.tir.Var:
            self.visit_var(expr)
        elif type_info == tvm.tir.FloatImm:
            self.visit_float_imm(expr)
        elif type_info == tvm.tir.IntImm:
            self.visit_int_imm(expr)
        elif type_info == tvm.tir.UIntImm:
            self.visit_uint_imm(expr)
        elif type_info == tvm.tir.StringImm:
            self.visit_string_imm(expr)
        elif type_info == tvm.tir.Add:
            self.visit_add(expr)
        elif type_info == tvm.tir.Sub:
            self.visit_sub(expr)
        elif type_info == tvm.tir.Mul:
            self.visit_mul(expr)
        elif type_info == tvm.tir.Div:
            self.visit_div(expr)
        elif type_info == tvm.tir.Mod:
            self.visit_mod(expr)
        elif type_info == tvm.tir.And:
            self.visit_and(expr)
        elif type_info == tvm.tir.Or:
            self.visit_or(expr)
        elif type_info == tvm.tir.Not:
            self.visit_not(expr)
        elif type_info == tvm.tir.Cast:
            self.visit_cast(expr)
        elif type_info == tvm.tir.EQ:
            self.visit_eq(expr)
        elif type_info == tvm.tir.NE:
            self.visit_ne(expr)
        elif type_info == tvm.tir.LT:
            self.visit_lt(expr)
        elif type_info == tvm.tir.LE:
            self.visit_le(expr)
        elif type_info == tvm.tir.GT:
            self.visit_gt(expr)
        elif type_info == tvm.tir.GE:
            self.visit_ge(expr)
        elif type_info == tvm.tir.Min:
            self.visit_min(expr)
        elif type_info == tvm.tir.Max:
            self.visit_max(expr)
        else:
            raise RuntimeError("Unsupported type: %s" % type_info)
        
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
