from __future__ import absolute_import

import tvm


class ExprVisitor(object):
    def visit(self, expr):
        type_info = type(expr)
        if type_info == tvm.expr.Reduce:
            res = self.visit_reduce(expr)
        elif type_info == tvm.expr.Call:
            res = self.visit_call(expr)
        elif type_info == tvm.expr.Var:
            res = self.visit_var(expr)
        elif type_info == tvm.expr.FloatImm:
            res = self.visit_float_imm(expr)
        elif type_info == tvm.expr.IntImm:
            res = self.visit_int_imm(expr)
        elif type_info == tvm.expr.UIntImm:
            res = self.visit_uint_imm(expr)
        elif type_info == tvm.expr.StringImm:
            res = self.visit_string_imm(expr)
        elif type_info == tvm.expr.Add:
            res = self.visit_add(expr)
        elif type_info == tvm.expr.Sub:
            res = self.visit_sub(expr)
        elif type_info == tvm.expr.Mul:
            res = self.visit_mul(expr)
        elif type_info == tvm.expr.Div:
            res = self.visit_div(expr)
        elif type_info == tvm.expr.Mod:
            res = self.visit_mod(expr)
        elif type_info == tvm.expr.And:
            res = self.visit_and(expr)
        elif type_info == tvm.expr.Or:
            res = self.visit_or(expr)
        elif type_info == tvm.expr.Not:
            res = self.visit_not(expr)
        elif type_info == tvm.expr.Cast:
            res = self.visit_cast(expr)
        elif type_info == tvm.expr.EQ:
            res = self.visit_eq(expr)
        elif type_info == tvm.expr.NE:
            res = self.visit_ne(expr)
        elif type_info == tvm.expr.LT:
            res = self.visit_lt(expr)
        elif type_info == tvm.expr.LE:
            res = self.visit_le(expr)
        elif type_info == tvm.expr.GT:
            res = self.visit_gt(expr)
        elif type_info == tvm.expr.GE:
            res = self.visit_ge(expr)
        elif type_info == tvm.expr.Min:
            res = self.visit_min(expr)
        elif type_info == tvm.expr.Max:
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
