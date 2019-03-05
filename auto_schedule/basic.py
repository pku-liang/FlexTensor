import tvm
import math
from collections import deque
from auto_schedule.utils import to_int, to_tuple, split_part_names


NODE_LST = []
NODE_TABLE = dict()


class ASTNode(object):
    def __init__(self, children_number):
        self.parent = None
        self.children_number = children_number
        self.children = [None for i in range(children_number)]

    def set_parent(self, p):
        if self.parent:
            raise RuntimeWarning("Changing the parent manually one more time")
        self.parent = p

    def set_child(self, c, pos):
        assert 0 <= pos < self.children_number
        self.children[pos] = c
        if c is not None:
            c.parent = self

    def child_position(self, child):
        assert child in self.children
        return self.children.index(child)

    def clone(self):
        raise NotImplementedError()

    def print(self, indent):
        print(indent, str(self.__class__).split(".")[-1], " [", self.children_number, "]")


class ForNode(ASTNode):
    def __init__(self, iter_var, body):
        super(ForNode, self).__init__(4)
        if iter_var is None and body is None:
            return
        beg = to_int(iter_var.dom.min)
        assert beg == 0     # if not zero, need to simplify it before
        extent = to_int(iter_var.dom.extent)
        self.set_child(VarNode(iter_var.var.name), 0)
        # self.set_child(IntNode(beg), 1)   # no need, because beg must be zero
        self.set_child(IntNode(extent), 2)
        self.set_child(body, 3)

    def clone(self):
        clone_self = ForNode(None, None)
        return clone_self

    def __eq__(self, other):
        return isinstance(other, ForNode) and self.children[0].var_name == other.children[0].var_name and self.children[
            2].value == self.children[2].value

    def __ne__(self, other):
        return not self == other


NODE_LST.append(ForNode)


class SeqNode(ASTNode):
    def __init__(self, l, r):
        super(SeqNode, self).__init__(2)
        self.set_child(l, 0)
        self.set_child(r, 1)

    def clone(self):
        clone_self = SeqNode(None, None)
        return clone_self


NODE_LST.append(SeqNode)


class BchNode(ASTNode):
    def __init__(self, c, body_true, body_false):
        super(BchNode, self).__init__(3)
        self.set_child(c, 0)
        self.set_child(body_true, 1)
        self.set_child(body_false, 2)

    def clone(self):
        clone_self = BchNode(None, None, None)
        return clone_self


NODE_LST.append(BchNode)


class FltNode(ASTNode):
    def __init__(self, val):
        super(FltNode, self).__init__(0)
        if not isinstance(val, float):
            raise ValueError("value type should be float, but got {}".format(type(val)))
        self.value = val

    def clone(self):
        clone_self = FltNode(self.value)
        return clone_self

    def print(self, indent):
        print(indent, self.value, "(float)")


NODE_LST.append(FltNode)


class IntNode(ASTNode):
    def __init__(self, val):
        super(IntNode, self).__init__(0)
        if not isinstance(val, int):
            raise ValueError("value type should be int, but got {}".format(type(val)))
        self.value = val
        
    def clone(self):
        clone_self = IntNode(self.value)
        return clone_self

    def print(self, indent):
        print(indent, self.value, "(int)")


NODE_LST.append(IntNode)


class BinNode(ASTNode):
    def __init__(self, l, r):
        super(BinNode, self).__init__(2)
        self.set_child(l, 0)
        self.set_child(r, 1)
        
    def clone(self):
        clone_self = self.__class__(None, None)
        return clone_self
        

class AddNode(BinNode):
    def __init__(self, l, r):
        super(AddNode, self).__init__(l, r)


NODE_LST.append(AddNode)


class SubNode(BinNode):
    def __init__(self, l, r):
        super(SubNode, self).__init__(l, r)


NODE_LST.append(SubNode)


class MulNode(BinNode):
    def __init__(self, l, r):
        super(MulNode, self).__init__(l, r)


NODE_LST.append(MulNode)


class DivNode(BinNode):
    def __init__(self, l, r):
        super(DivNode, self).__init__(l, r)


NODE_LST.append(DivNode)


class AndNode(BinNode):
    def __init__(self, l, r):
        super(AndNode, self).__init__(l, r)


NODE_LST.append(AndNode)


class LorNode(BinNode):
    def __init__(self, l, r):
        super(LorNode, self).__init__(l, r)


NODE_LST.append(LorNode)


class NotNode(ASTNode):
    def __init__(self, l):
        super(NotNode, self).__init__(1)
        self.set_child(l, 0)
        
    def clone(self):
        clone_self = NotNode(None)
        return clone_self


NODE_LST.append(NotNode)


class IncNode(BinNode):
    def __init__(self, l, r):
        super(IncNode, self).__init__(l, r)


NODE_LST.append(IncNode)


class EqlNode(BinNode):
    def __init__(self, l, r):
        super(EqlNode, self).__init__(l, r)


NODE_LST.append(EqlNode)


class EeqNode(BinNode):
    def __init__(self, l, r):
        super(EeqNode, self).__init__(l, r)


NODE_LST.append(EeqNode)


class NeqNode(BinNode):
    def __init__(self, l, r):
        super(NeqNode, self).__init__(l, r)


NODE_LST.append(NeqNode)


class LeqNode(BinNode):
    def __init__(self, l, r):
        super(LeqNode, self).__init__(l, r)


NODE_LST.append(LeqNode)


class LesNode(BinNode):
    def __init__(self, l, r):
        super(LesNode, self).__init__(l, r)


NODE_LST.append(LesNode)


class GeqNode(BinNode):
    def __init__(self, l, r):
        super(GeqNode, self).__init__(l, r)


NODE_LST.append(GeqNode)


class GreNode(BinNode):
    def __init__(self, l, r):
        super(GreNode, self).__init__(l, r)


NODE_LST.append(GreNode)


class VstNode(ASTNode):
    def __init__(self, op, indice, shape):
        super(VstNode, self).__init__(2)
        if op is None and indice is None and shape is None:
            return
        assert len(indice) > 0 and len(indice) == len(shape)
        cur = indice[0]
        p = 1
        end = len(indice)
        while p < end:
            int_node = IntNode(to_int(shape[p]))
            mul_node = MulNode(cur, int_node)
            add_node = AddNode(mul_node, indice[p])
            cur = add_node
            p += 1
        self.set_child(OprNode(op), 0)
        self.set_child(cur, 1)

    def clone(self):
        clone_self = VstNode(None, None, None)
        return clone_self


NODE_LST.append(VstNode)


class VarNode(ASTNode):
    def __init__(self, var_name):
        super(VarNode, self).__init__(0)
        self.var_name = var_name

    def clone(self):
        clone_self = VarNode(self.var_name)
        return clone_self

    def print(self, indent):
        print(indent, self.var_name)


NODE_LST.append(VarNode)


class OprNode(ASTNode):
    def __init__(self, op):
        super(OprNode, self).__init__(0)
        self.op = op

    def clone(self):
        clone_self = OprNode(self.op)
        return clone_self

    def print(self, indent):
        print(indent, self.op)


NODE_LST.append(OprNode)


class NopNode(ASTNode):
    def __init__(self, op, tree_root):
        super(NopNode, self).__init__(1)
        self.set_child(tree_root, 0)
        self.op = op

    def clone(self):
        clone_self = NopNode(self.op, None)
        return clone_self


NODE_LST.append(NopNode)


# from Node class to label
for i, NODE in enumerate(NODE_LST):
    NODE_TABLE[NODE] = i


class AST(object):
    def __init__(self):
        self.root = None

    def set_root(self, new_root):
        assert isinstance(new_root, ASTNode)
        if self.root:
            raise RuntimeWarning("Changing tree root manually one more time")
        self.root = new_root

    def is_empty(self):
        return self.root is None

    def split(self, iter_var_name, factor):
        assert isinstance(self.root, NopNode)
        ret = []
        collect_nodes = []
        recursive_collect_iter_var(iter_var_name, self.root, collect_nodes)
        new_iter_var_names = split_part_names(iter_var_name, 2)
        for node in collect_nodes:
            # the first pass to find where the for node is
            if isinstance(node.parent, ForNode):
                pnode = node.parent
                ppnode = pnode.parent
                extent = pnode.children[2].value
                body = pnode.children[3]
                inner_node = ForNode(None, None)
                outer_node = ForNode(None, None)
                inner = factor
                outer = math.ceil(extent / factor)
                inner_node.set_child(VarNode(new_iter_var_names[1]), 0)
                inner_node.set_child(IntNode(inner), 2)
                inner_node.set_child(body, 3)
                outer_node.set_child(VarNode(new_iter_var_names[0]), 0)
                outer_node.set_child(IntNode(outer), 2)
                outer_node.set_child(inner_node, 3)
                if ppnode is not None:
                    position = ppnode.child_position(pnode)
                    ppnode.set_child(outer_node, position)
                ret.extend([outer_node, inner_node])
        for node in collect_nodes:
            # the second pass update the indices
            if not isinstance(node.parent, ForNode):
                pnode = node.parent
                position = pnode.child_position(node)
                inner_node = VarNode(new_iter_var_names[1])
                outer_node = VarNode(new_iter_var_names[0])
                factor_node = IntNode(factor)
                mul_node = MulNode(outer_node, factor_node)
                add_node = AddNode(mul_node, inner_node)
                pnode.set_child(add_node, position)
        return ret

    def reorder(self, iter_var_name_lst, loop_msg):
        for_node_lst = []
        for iter_var_name in iter_var_name_lst:
            for_nodes = []
            recursive_collect_for_node(iter_var_name, self.root, for_nodes)
            assert len(for_nodes) == 1  # one iter_var should corresponding to one for loop
            for_node_lst.append(for_nodes[0])
        all_node_lst = []
        p_last_spatial = None
        for iter_var_name in loop_msg.iter_var_names:
            for_node = loop_msg.get_for_node(iter_var_name)
            all_node_lst.append(for_node)
            if not loop_msg.is_reduce(iter_var_name):
                p_last_spatial = for_node
        spatial_end_body = p_last_spatial.children[3]
        end_body = all_node_lst[-1].children[3]
        index_lst = []

        for node in for_node_lst:
            index_lst.append(all_node_lst.index(node))
        index_lst = list(sorted(index_lst))
        for i, index in enumerate(index_lst):
            all_node_lst[index] = for_node_lst[i]
        later_last_spatial = None
        for node in all_node_lst:
            if not loop_msg.is_reduce(node.children[0].var_name):
                later_last_spatial = node
        # link spatial for nodes
        p = 0
        while all_node_lst[p] != later_last_spatial:
            all_node_lst[p].set_child(all_node_lst[p + 1], 3)
            p += 1
        p += 1
        if p < len(all_node_lst):   # has reduce for nodes
            assert isinstance(spatial_end_body, SeqNode)
            later_last_spatial.set_child(spatial_end_body, 3)
            spatial_end_body.set_child(all_node_lst[p], 1)
            while p < len(all_node_lst) - 1:
                all_node_lst[p].set_child(all_node_lst[p + 1], 3)
                p += 1
            all_node_lst[p].set_child(end_body, 3)
        else:
            if isinstance(spatial_end_body, SeqNode):
                later_last_spatial.set_child(spatial_end_body, 3)
                spatial_end_body.set_child(end_body, 1)
            else:
                later_last_spatial.set_child(end_body, 3)
        self.root.set_child(all_node_lst[0], 0)

    def clone(self):
        loop_msg = {}
        clone_root = self._recursive_clone(self.root, loop_msg)
        clone_self = AST()
        clone_self.set_root(clone_root)
        return clone_self, loop_msg

    def _recursive_clone(self, node, loop_msg):
        if node is None:
            return None
        assert isinstance(node, ASTNode)
        clone_node = node.clone()
        for i, child in enumerate(node.children):
            clone_result = self._recursive_clone(child, loop_msg)
            clone_node.set_child(clone_result, i)
        if isinstance(clone_node, ForNode):
            loop_msg[clone_node.children[0].var_name] = clone_node
        return clone_node

    def flatten(self, loop_msg_dict, op=None):
        flattened = []
        if op is None:
            cur_op = None
            self._recursive_flatten(cur_op, self.root, loop_msg_dict, flattened)
        else:
            assert isinstance(op, AST)
            cur_op = op.root.op
            self._recursive_flatten(cur_op, op.root, loop_msg_dict, flattened)
        return flattened

    def _recursive_flatten(self, op, root, loop_msg_dict, flattened):
        if root is None:
            return
        assert isinstance(root, ASTNode)
        if isinstance(root, NopNode):
            op = root.op
        flattened.append(root)
        if isinstance(root, VarNode) and not isinstance(root.parent, ForNode):
            for_node = loop_msg_dict[op].for_node_dict[root.var_name]
            position = loop_msg_dict[op].iter_var_names.index(root.var_name)
            flattened.append(IntNode(position))
            flattened.append(IntNode(for_node.children[2].value))
        for child in root.children:
            self._recursive_flatten(op, child, loop_msg_dict, flattened)

    def collect_iter_var_feature(self, loop_msg, sub_tree, iter_var_name):
        feature = []
        self._recursive_collect_feature(sub_tree.root, loop_msg, iter_var_name, feature)
        return feature

    def _recursive_collect_feature(self, root, loop_msg, iter_var_name, feature):
        if root is None:
            return False
        if isinstance(root, VarNode) and root.var_name == iter_var_name:
            feature.append(root)
            for_node = loop_msg.for_node_dict[iter_var_name]
            feature.append(IntNode(for_node.children[2].value))
            feature.append(IntNode(loop_msg.iter_var_names.index(iter_var_name)))
            return True
        flag = False
        for child in root.children:
            if self._recursive_collect_feature(child, loop_msg, iter_var_name, feature):
                flag = True
        if flag:
            for child in root.children:
                if child is not None:
                    feature.append(child)
            return True and not isinstance(root, (ForNode, SeqNode, BchNode, VstNode))
        return False

    @classmethod
    def concat_subtree(cls, tree_lst):
        new_lst = []
        for tree in tree_lst:
            if not tree.is_empty():
                new_lst.append(tree)
        tree_lst = new_lst
        if not tree_lst:
            return AST()
        cur = tree_lst[0]
        p = 1
        length = len(tree_lst)
        while p < length:
            seq_node = SeqNode(cur.root, tree_lst[p].root)
            cur = AST()
            cur.set_root(seq_node)
            p += 1
        return cur

    def print(self):
        print("**********************************")
        self._print(self.root, 0)
        print("**********************************")

    def _print(self, root, level):
        indent = "|  " * max(level - 1, 0) + "|--" if level > 0 else ""
        if root is None:
            print(indent, "None")
            return
        root.print(indent)
        for child in root.children:
            self._print(child, level+1)


class LoopMessage(object):
    def __init__(self):
        self.iter_var_names = []
        self.iter_vars = {}
        self.for_node_dict = {}
        self.reduce_set = set()

    def append(self, iter_var, for_node, reduce=False):
        self.iter_var_names.append(iter_var.var.name)
        self.iter_vars[iter_var.var.name] = iter_var
        if reduce:
            self.reduce_set.add(iter_var.var.name)
        self.for_node_dict[iter_var.var.name] = for_node

    def get_iter_var(self, iter_var_name):
        assert iter_var_name in self.iter_var_names
        return self.iter_vars[iter_var_name]

    def get_for_node(self, iter_var_name):
        assert iter_var_name in self.iter_var_names
        return self.for_node_dict[iter_var_name]

    def get_iter_var_extent(self, iter_var_name):
        assert iter_var_name in self.iter_var_names
        return self.for_node_dict[iter_var_name].children[2].value

    def is_reduce(self, iter_var_name):
        assert iter_var_name in self.iter_var_names
        return iter_var_name in self.reduce_set

    def split(self, iter_var_name, new_iter_var_lst, new_for_nodes):
        assert iter_var_name in self.iter_var_names and len(new_iter_var_lst) == len(new_for_nodes)
        p = self.iter_var_names.index(iter_var_name)
        del self.iter_var_names[p]
        del self.iter_vars[iter_var_name]
        del self.for_node_dict[iter_var_name]

        new_iter_var_names = split_part_names(iter_var_name, len(new_iter_var_lst))
        for i, new_iter_var in enumerate(new_iter_var_lst):
            self.iter_var_names.insert(p + i, new_iter_var_names[i])
            self.iter_vars[new_iter_var_names[i]] = new_iter_var
            self.for_node_dict[new_iter_var_names[i]] = new_for_nodes[i]

        if iter_var_name in self.reduce_set:
            self.reduce_set.remove(iter_var_name)
            for new_iter_var_name in new_iter_var_names:
                self.reduce_set.add(new_iter_var_name)

    def replace(self, iter_var_name, new_iter_var):
        assert iter_var_name in self.iter_vars
        self.iter_vars[iter_var_name] = new_iter_var

    def update_iter_var_lst_only(self, iter_var_name, new_iter_var_lst):
        assert iter_var_name in self.iter_var_names
        p = self.iter_var_names.index(iter_var_name)
        del self.iter_var_names[p]
        del self.iter_vars[iter_var_name]
        del self.for_node_dict[iter_var_name]

        new_iter_var_names = split_part_names(iter_var_name, len(new_iter_var_lst))
        for i, new_iter_var in enumerate(new_iter_var_lst):
            self.iter_var_names.insert(p + i, new_iter_var_names[i])
            self.iter_vars[new_iter_var_names[i]] = new_iter_var

    def update_for_node_dict_once(self, reduce_msg, for_node_loop_msg):
        self.reduce_set = reduce_msg.copy()
        self.for_node_dict = for_node_loop_msg
        for name, node in for_node_loop_msg.items():
            print(name, node.children[0].var_name)

    def reorder(self, iter_var_name_lst):
        index_lst = []
        for iter_var_name in iter_var_name_lst:
            index_lst.append(self.iter_var_names.index(iter_var_name))
        index_lst = list(sorted(index_lst))
        for i, index in enumerate(index_lst):
            self.iter_var_names[index] = iter_var_name_lst[i]

    def clone(self):
        clone_self = LoopMessage()
        for name in self.iter_var_names:
            if name in self.reduce_set:
                clone_self.append(self.iter_vars[name], self.for_node_dict[name], reduce=True)
            else:
                clone_self.append(self.iter_vars[name], self.for_node_dict[name], reduce=False)
        return clone_self

    def __str__(self):
        ret = "loop order " + str(self.iter_var_names)
        return ret


class Reference(object):
    def __init__(self):
        self.ref = dict()

    def add(self, op, tree):
        assert op not in self.ref and isinstance(tree, AST)
        self.ref[op] = tree

    def set(self, op, tree):
        assert op in self.ref and isinstance(tree, AST)
        self.ref[op] = tree

    def get(self, op):
        assert op in self.ref
        return self.ref[op]

    def remove(self, op):
        assert op in self.ref
        del self.ref[op]

    def clone(self):
        clone_self = Reference()
        for op, tree in self.ref.items():
            clone_self.add(op, tree)
        return clone_self

    def print(self):
        for op, tree in self.ref.items():
            print("sub-tree for operation ", op, ":")
            tree.print()


class Action(object):
    def __init__(self, op):
        self.op = op

    def clone(self):
        raise NotImplementedError()

    def apply(self, *args, **kwargs):
        raise NotImplementedError()

    def to_json(self, *args):
        raise NotImplementedError()


class Split(Action):
    def __init__(self, op, iter_var_name, factor, reduce=False):
        super(Split, self).__init__(op)
        self.iter_var_name = iter_var_name
        self.factor = factor
        self.reduce = reduce

    def clone(self):
        clone_self = Split(self.op, self.iter_var_name, self.factor, self.reduce)
        return clone_self

    def apply(self, env):
        loop_msg = env.loop_msg_dict[self.op]
        iter_var = loop_msg.get_iter_var(self.iter_var_name)
        outer, inner = env.sch[self.op].split(iter_var, factor=self.factor)
        return [outer, inner]

    def to_json(self, op_order_dict):
        return {
            "type": "split",
            "op": op_order_dict[self.op],
            "axis": self.iter_var_name,
            "factor": self.factor,
            "reduce": self.reduce
                }

    def __str__(self):
        ret = "split(" + str(self.op) + ", " + self.iter_var_name + ", " + str(self.factor) + ")"
        return ret


class Reorder(Action):
    def __init__(self, op, iter_var_name_lst):
        super(Reorder, self).__init__(op)
        self.iter_var_name_lst = iter_var_name_lst

    def clone(self):
        clone_self = Reorder(self.op, self.iter_var_name_lst)
        return clone_self

    def apply(self, env):
        loop_msg = env.loop_msg_dict[self.op]
        iter_var_lst = []
        for iter_var_name in self.iter_var_name_lst:
            iter_var_lst.append(loop_msg.get_iter_var(iter_var_name))
        env.sch[self.op].reorder(*iter_var_lst)
        return self.iter_var_name_lst

    def __str__(self):
        ret = "reorder(" + str(self.op) + ", " + ", ".join(self.iter_var_name_lst) + ")"
        return ret

    def to_json(self, op_order_dict):
        return {
            "type": "reorder",
            "op": op_order_dict[self.op],
            "axis_lst": self.iter_var_name_lst
        }


class Fuse(Action):
    def __init__(self, op, iter_var_name_lst):
        super(Fuse, self).__init__(op)
        self.iter_var_name_lst = iter_var_name_lst

    def clone(self):
        clone_self = Fuse(self.op, self.iter_var_name_lst)
        return clone_self

    def apply(self, env):
        loop_msg = env.loop_msg_dict[self.op]
        iter_var_lst = []
        for iter_var_name in self.iter_var_name_lst:
            iter_var = loop_msg.get_iter_var(iter_var_name)
            iter_var_lst.append(iter_var)
        fused_iter_var = env.sch[self.op].fuse(*iter_var_lst)
        return fused_iter_var

    def __str__(self):
        ret = "fuse(" + str(self.op) + ", ".join(self.iter_var_name_lst) + ")"
        return ret

    def to_json(self, op_order_dict):
        return {
            "type": "fuse",
            "op": op_order_dict[self.op],
            "axis_lst": self.iter_var_name_lst
        }


class ComputeAt(Action):
    def __init__(self, op, next_op, iter_var_name):
        super(ComputeAt, self).__init__(op)
        self.next_op = next_op
        self.iter_var_name = iter_var_name

    def clone(self):
        clone_self = ComputeAt(self.op, self.next_op, self.iter_var_name)
        return clone_self

    def apply(self, env):
        next_loop_msg = env.loop_msg_dict[self.next_op]
        iter_var = next_loop_msg.get_iter_var(self.iter_var_name)
        env.sch[self.op].compute_at(env.sch[self.next_op], iter_var)

    def __str__(self):
        ret = "compute_at(" + str(self.op) + ", " + str(self.next_op) + ", " + self.iter_var_name + ")"
        return ret

    def to_json(self, op_order_dict):
        return {
            "type": "compute_at",
            "op": op_order_dict[self.op],
            "next_op": self.next_op,
            "axis": self.iter_var_name
        }


class ComputeInline(Action):
    def __init__(self, op):
        super(ComputeInline, self).__init__(op)

    def clone(self):
        clone_self = ComputeInline(self.op)
        return clone_self

    def apply(self, env):
        env.sch[self.op].compute_inline()

    def __str__(self):
        ret = "compute_inline(" + str(self.op) + ")"
        return ret

    def to_json(self, op_order_dict):
        return {
            "type": "compute_inline",
            "op": op_order_dict[self.op],
        }


class Parallel(Action):
    def __init__(self, op, iter_var_name):
        super(Parallel, self).__init__(op)
        self.iter_var_name = iter_var_name

    def clone(self):
        clone_self = Parallel(self.op, self.iter_var_name)
        return clone_self

    def apply(self, env):
        loop_msg = env.loop_msg_dict[self.op]
        iter_var = loop_msg.get_iter_var(self.iter_var_name)
        env.sch[self.op].parallel(iter_var)

    def __str__(self):
        ret = "parallel(" + str(self.op) + ", " + self.iter_var_name + ")"
        return ret

    def to_json(self, op_order_dict):
        return {
            "type": "parallel",
            "op": op_order_dict[self.op],
            "axis": self.iter_var_name
        }


class Unroll(Action):
    def __init__(self, op, iter_var_name):
        super(Unroll, self).__init__(op)
        self.iter_var_name = iter_var_name

    def clone(self):
        clone_self = Unroll(self.op, self.iter_var_name)
        return clone_self

    def apply(self, env):
        loop_msg = env.loop_msg_dict[self.op]
        iter_var = loop_msg.get_iter_var(self.iter_var_name)
        env.sch[self.op].unroll(iter_var)

    def __str__(self):
        ret = "unroll(" + str(self.op) + ", " + self.iter_var_name + ")"
        return ret

    def to_json(self, op_order_dict):
        return {
            "type": "unroll",
            "op": op_order_dict[self.op],
            "axis": self.iter_var_name
        }


class Vectorize(Action):
    def __init__(self, op, iter_var_name):
        super(Vectorize, self).__init__(op)
        self.iter_var_name = iter_var_name

    def clone(self):
        clone_self = Vectorize(self.op, self.iter_var_name)
        return clone_self

    def apply(self, env):
        loop_msg = env.loop_msg_dict[self.op]
        iter_var = loop_msg.get_iter_var(self.iter_var_name)
        env.sch[self.op].vectorize(iter_var)

    def __str__(self):
        ret = "vectorize(" + str(self.op) + ", " + self.iter_var_name + ")"
        return ret

    def to_json(self, op_order_dict):
        return {
            "type": "vectorize",
            "op": op_order_dict[self.op],
            "axis": self.iter_var_name
        }


class Bind(Action):
    def __init__(self, op):
        super(Bind, self).__init__(op)

    def clone(self):
        pass

    def apply(self, *args, **kwargs):
        pass


class CacheRead(Action):
    def __init__(self, op):
        super(CacheRead, self).__init__(op)

    def clone(self):
        pass

    def apply(self, *args, **kwargs):
        pass


class CacheWrite(Action):
    def __init__(self, op):
        super(CacheWrite, self).__init__(op)

    def clone(self):
        pass

    def apply(self, *args, **kwargs):
        pass


class Environment(object):
    def __init__(self, compute_name, args, ops):
        self.compute_name = compute_name
        self.args = args
        self.ops = ops
        self.sch = None
        self.tree = None
        self.ref = None
        self.initial_loop_msg_dict = None
        self.loop_msg_dict = dict()
        self.action_lst = []

    def initialized(self):
        return isinstance(self.tree, AST) and isinstance(self.ref, Reference) \
            and self.loop_msg_dict and self.sch and self.initial_loop_msg_dict

    def set_schedule(self, sch):
        if self.sch:
            raise RuntimeWarning("Changing schedule manually one more time")
        self.sch = sch

    def set_tree(self, tree):
        assert isinstance(tree, AST)
        if self.tree:
            raise RuntimeWarning("Changing tree manually one more time")
        self.tree = tree

    def set_ref(self, ref):
        assert isinstance(ref, Reference)
        if self.ref:
            raise RuntimeWarning("Changing reference manually one more time")
        self.ref = ref

    def add_loop_msg(self, op, loop_msg):
        assert op not in self.loop_msg_dict and isinstance(loop_msg, LoopMessage)
        self.loop_msg_dict[op] = loop_msg

    def set_loop_msg(self, op, loop_msg):
        assert op in self.loop_msg_dict and isinstance(loop_msg, LoopMessage)
        self.loop_msg_dict[op] = loop_msg

    def remove_loop_msg(self, op):
        assert op in self.loop_msg_dict
        del self.loop_msg_dict[op]

    def add_action(self, action):
        assert isinstance(action, Action)
        self.action_lst.append(action)

    def take_action(self, action, no_clone=True):
        if no_clone:
            clone_self = self
        else:
            clone_self = self.clone()
        clone_self.add_action(action)
        if isinstance(action, Split):
            clone_self.split(action)
        elif isinstance(action, Reorder):
            clone_self.reorder(action)
        elif isinstance(action, ComputeAt):
            clone_self.compute_at(action)
        elif isinstance(action, ComputeInline):
            clone_self.compute_inline(action)
        elif isinstance(action, Parallel):
            clone_self.parallel(action)
        elif isinstance(action, Unroll):
            clone_self.unroll(action)
        elif isinstance(action, Vectorize):
            clone_self.vectorize(action)
        elif isinstance(action, Bind):
            clone_self.bind(action)
        elif isinstance(action, CacheRead):
            clone_self.cache_read(action)
        elif isinstance(action, CacheWrite):
            clone_self.cache_write(action)
        else:
            raise NotImplementedError("No support for {} in tvm".format(str(action)))
        return clone_self

    def get_iter_var_extent(self, op, iter_var_name):
        assert isinstance(iter_var_name, str)
        return self.loop_msg_dict[op].get_iter_var_extent(iter_var_name)

    def clone(self):
        clone_self = build_environment(self.compute_name, self.args, self.ops)
        for act in self.action_lst:
            clone_self.add_action(act)
            if isinstance(act, Split):
                clone_self.split(act)
            elif isinstance(act, Reorder):
                clone_self.reorder(act)
            elif isinstance(act, ComputeAt):
                clone_self.compute_at(act)
            elif isinstance(act, ComputeInline):
                clone_self.compute_inline(act)
            elif isinstance(act, Parallel):
                clone_self.parallel(act)
            elif isinstance(act, Unroll):
                clone_self.unroll(act)
            elif isinstance(act, Vectorize):
                clone_self.vectorize(act)
            elif isinstance(act, Bind):
                clone_self.bind(act)
            elif isinstance(act, CacheRead):
                clone_self.cache_read(act)
            elif isinstance(act, CacheWrite):
                clone_self.cache_write(act)
            else:
                raise NotImplementedError("No support for {} in tvm".format(str(act)))
        return clone_self

    def split(self, action):
        op = action.op
        iter_var_name = action.iter_var_name
        factor = action.factor
        sub_tree = self.ref.get(op)
        new_for_node_lst = sub_tree.split(iter_var_name, factor)
        new_iter_var_lst = action.apply(self)
        self.loop_msg_dict[op].split(iter_var_name, new_iter_var_lst, new_for_node_lst)
        return self

    def reorder(self, action):
        op = action.op
        iter_var_name_lst = action.iter_var_name_lst
        sub_tree = self.ref.get(op)
        sub_tree.reorder(iter_var_name_lst, self.loop_msg_dict[op])
        action.apply(self)
        self.loop_msg_dict[op].reorder(iter_var_name_lst)
        return self

    def compute_at(self, action):
        op = action.op
        next_op = action.next_op
        iter_var_name = action.iter_var_name
        sub_tree = self.ref.get(op)
        next_sub_tree = self.ref.get(next_op)
        sub_tree.compute_at(next_sub_tree, iter_var_name)

    def compute_inline(self, action):
        raise NotImplementedError()

    def parallel(self, action):
        raise NotImplementedError()

    def unroll(self, action):
        raise NotImplementedError()

    def vectorize(self, action):
        raise NotImplementedError()

    def bind(self, action):
        raise NotImplementedError()

    def cache_read(self, action):
        raise NotImplementedError()

    def cache_write(self, action):
        raise NotImplementedError()

    def flatten_tree(self, op=None):
        if isinstance(op, tvm.tensor.ComputeOp):
            return self.tree.flatten(self.loop_msg_dict, self.ref.get(op))
        else:
            return self.tree.flatten(self.loop_msg_dict, None)

    def collect_iter_var_feature(self, op, axis):
        loop_msg = self.loop_msg_dict[op]
        sub_tree = self.ref.get(op)
        return self.tree.collect_iter_var_feature(loop_msg, sub_tree, axis)

    def to_json(self, op_order_dict):
        return {
            "compute_name": self.compute_name,
            "args": str(self.args),
            "actions": [act.to_json(op_order_dict) for act in self.action_lst]
        }

    def print(self, tree=False):
        if tree:
            print("the ast tree:")
            self.tree.print()
        self.ref.print()
        for op, loop_msg in self.loop_msg_dict.items():
            print("loop message for operation ", op, " is: ", str(loop_msg))
            var_name_lst = []
            for var_name, for_node in loop_msg.for_node_dict.items():
                var_name_lst.append(var_name)
            print(str(var_name_lst))


def recursive_collect_iter_var(iter_var_name, root, result):
    if root is None:
        return
    if isinstance(root, VarNode) and root.var_name == iter_var_name:
        result.append(root)
    for child in root.children:
        if not isinstance(child, NopNode):
            recursive_collect_iter_var(iter_var_name, child, result)


def recursive_collect_for_node(iter_var_name, root, result):
    if root is None:
        return
    if isinstance(root, ForNode) and root.children[0].var_name == iter_var_name:
        result.append(root)
    for child in root.children:
        if not isinstance(child, NopNode):
            recursive_collect_for_node(iter_var_name, child, result)


def recursive_collect_all_for_nodes(root, result):
    if root is None:
        return
    if isinstance(root, ForNode):
        result.append(root)
    for child in root.children:
        if not isinstance(child, NopNode):
            recursive_collect_all_for_nodes(child, result)


def bfs_traverse(ops, reverse=True):
    ret = []
    visited = set()
    q = deque()
    for op in ops:
        q.append(op)
        visited.add(op)
    while q:
        cur = q.popleft()
        ret.append(cur)
        for t in cur.input_tensors:
            if t.op not in visited:
                q.append(t.op)
                visited.add(t.op)
    if reverse:
        ret = list(reversed(ret))   # from input to output
    return ret


def build_environment(compute_name, args, ops):
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    s = tvm.create_schedule(ops)
    op_lst = bfs_traverse(ops)
    env = Environment(compute_name, args, ops)
    sub_tree_lst = []
    ref_msg = Reference()
    for op in op_lst:
        subtree, loop_msg = build_single_operation_tree(op, ref_msg)
        if subtree.is_empty():  # ignore empty tree
            continue
        env.add_loop_msg(op, loop_msg)
        sub_tree_lst.append(subtree)
    tree = AST.concat_subtree(sub_tree_lst)
    env.set_schedule(s)
    env.set_tree(tree)
    env.set_ref(ref_msg)
    env.initial_loop_msg_dict = env.loop_msg_dict
    return env


def build_single_operation_tree(op, ref_msg):
    tree = AST()
    loop_msg = LoopMessage()
    # empty tree for placeholder
    if isinstance(op, tvm.tensor.PlaceholderOp):
        # do not record empty tree
        return tree, loop_msg
    # subtree for compute
    elif isinstance(op, tvm.tensor.ComputeOp):
        # the first part, loop generation
        loop_lst = []       # non-reduce axis
        inner_lst = []      # reduce axis
        for iter_var in op.axis:
            for_node = ForNode(iter_var, None)
            loop_lst.append(for_node)
            loop_msg.append(iter_var, for_node)       # record this iter_var
        for i, for_node in enumerate(loop_lst[:-1]):
            for_node.set_child(loop_lst[i + 1], 3)     # link ForNodes
        # the second part, reduce initialization
        if op.reduce_axis:          # if has reduce
            for iter_var in op.reduce_axis:
                for_node = ForNode(iter_var, None)
                inner_lst.append(for_node)
                loop_msg.append(iter_var, for_node, reduce=True)   # record this iter_var
            for i, for_node in enumerate(inner_lst[:-1]):
                for_node.set_child(inner_lst[i + 1], 3)     # link ForNodes
            sub_tree_lst = []
            # the init for reduce operation (assume zero initialization)
            for i in range(op.num_outputs):     # may be many outputs
                indice = []
                for for_node in loop_lst:
                    indice.append(VarNode(for_node.children[0].var_name))     # the iter vars
                vst_node = VstNode(op, indice, to_tuple(op.output(i).shape))
                if op.output(i).dtype == "float32":
                    eq_node = EqlNode(vst_node, FltNode(0.0))
                elif op.output(i).dtype == "int32":
                    eq_node = EqlNode(vst_node, IntNode(0))
                else:
                    raise ValueError(
                        "only 'float32' and 'int32' supported in tvm, but got {}".format(op.output(i).dtype))
                sub_tree = AST()
                sub_tree.set_root(eq_node)
                sub_tree_lst.append(sub_tree)
            left_sub_tree = AST.concat_subtree(sub_tree_lst)
            sub_tree_lst.clear()
            for i in range(op.num_outputs):
                indice = []
                for for_node in loop_lst:
                    indice.append(VarNode(for_node.children[0].var_name))     # the iter vars
                vst_node = VstNode(op, indice, to_tuple(op.output(i).shape))
                vst_tree = AST()
                vst_tree.set_root(vst_node)
                sub_tree = build_statement_tree(op, vst_tree, i, True)
                sub_tree_lst.append(sub_tree)
            right_sub_tree = AST.concat_subtree(sub_tree_lst)
            inner_lst[-1].set_child(right_sub_tree.root, 3)
            seq_node = SeqNode(left_sub_tree.root, inner_lst[0])
            loop_lst[-1].set_child(seq_node, 3)
        else:
            sub_tree_lst = []
            for i in range(op.num_outputs):
                indice = []
                for for_node in loop_lst:
                    indice.append(VarNode(for_node.children[0].var_name))     # the iter vars
                vst_node = VstNode(op.output(i), indice, to_tuple(op.output(i).shape))
                vst_tree = AST()
                vst_tree.set_root(vst_node)
                sub_tree = build_statement_tree(op, vst_tree, i, False)
                sub_tree_lst.append(sub_tree)
            right_sub_tree = AST.concat_subtree(sub_tree_lst)
            loop_lst[-1].set_child(right_sub_tree.root, 3)
        nop_node = NopNode(op, loop_lst[0])
        tree.set_root(nop_node)
        ref_msg.add(op, tree)
        return tree, loop_msg
    else:
        raise ValueError("operation not implemented in tvm: {}".format(str(op)))


def build_var_tree(var):
    tree = AST()
    var_node = VarNode(var.name)
    tree.set_root(var_node)
    return tree


def build_const_tree(const_expr):
    tree = AST()
    if const_expr.dtype == "float32":
        v_node = FltNode(const_expr.value)
    elif const_expr.dtype == "int32":
        v_node = IntNode(const_expr.value)
    else:
        raise ValueError("tvm only support 'float32' and 'int32' but got {}".format(const_expr.dtype))
    tree.set_root(v_node)
    return tree


def build_add_tree(add_expr):
    tree = AST()
    left_tree = build_expr_tree(add_expr.a)
    right_tree = build_expr_tree(add_expr.b)
    add_node = AddNode(left_tree.root, right_tree.root)
    tree.set_root(add_node)
    return tree


def build_sub_tree(sub_expr):
    tree = AST()
    left_tree = build_expr_tree(sub_expr.a)
    right_tree = build_expr_tree(sub_expr.b)
    sub_node = SubNode(left_tree.root, right_tree.root)
    tree.set_root(sub_node)
    return tree


def build_mul_tree(mul_expr):
    tree = AST()
    left_tree = build_expr_tree(mul_expr.a)
    right_tree = build_expr_tree(mul_expr.b)
    mul_node = MulNode(left_tree.root, right_tree.root)
    tree.set_root(mul_node)
    return tree


def build_div_tree(div_expr):
    tree = AST()
    left_tree = build_expr_tree(div_expr.a)
    right_tree = build_expr_tree(div_expr.b)
    div_node = DivNode(left_tree.root, right_tree.root)
    tree.set_root(div_node)
    return tree


def build_and_tree(and_expr):
    tree = AST()
    left_tree = build_expr_tree(and_expr.a)
    right_tree = build_expr_tree(and_expr.b)
    and_node = AndNode(left_tree.root, right_tree.root)
    tree.set_root(and_node)
    return tree


def build_or_tree(or_expr):
    tree = AST()
    left_tree = build_expr_tree(or_expr.a)
    right_tree = build_expr_tree(or_expr.b)
    or_node = LorNode(left_tree.root, right_tree.root)
    tree.set_root(or_node)
    return tree


def build_not_tree(not_expr):
    tree = AST()
    left_tree = build_expr_tree(not_expr.a)
    not_node = NotNode(left_tree.root)
    tree.set_root(not_node)
    return tree


def build_eq_tree(cmp_expr):
    tree = AST()
    left_tree = build_expr_tree(cmp_expr.a)
    right_tree = build_expr_tree(cmp_expr.b)
    cmp_node = EeqNode(left_tree.root, right_tree.root)
    tree.set_root(cmp_node)
    return tree


def build_ne_tree(cmp_expr):
    tree = AST()
    left_tree = build_expr_tree(cmp_expr.a)
    right_tree = build_expr_tree(cmp_expr.b)
    cmp_node = NeqNode(left_tree.root, right_tree.root)
    tree.set_root(cmp_node)
    return tree


def build_le_tree(cmp_expr):
    tree = AST()
    left_tree = build_expr_tree(cmp_expr.a)
    right_tree = build_expr_tree(cmp_expr.b)
    cmp_node = LeqNode(left_tree.root, right_tree.root)
    tree.set_root(cmp_node)
    return tree


def build_lt_tree(cmp_expr):
    tree = AST()
    left_tree = build_expr_tree(cmp_expr.a)
    right_tree = build_expr_tree(cmp_expr.b)
    cmp_node = LesNode(left_tree.root, right_tree.root)
    tree.set_root(cmp_node)
    return tree


def build_ge_tree(cmp_expr):
    tree = AST()
    left_tree = build_expr_tree(cmp_expr.a)
    right_tree = build_expr_tree(cmp_expr.b)
    cmp_node = GeqNode(left_tree.root, right_tree.root)
    tree.set_root(cmp_node)
    return tree


def build_gt_tree(cmp_expr):
    tree = AST()
    left_tree = build_expr_tree(cmp_expr.a)
    right_tree = build_expr_tree(cmp_expr.b)
    cmp_node = GreNode(left_tree.root, right_tree.root)
    tree.set_root(cmp_node)
    return tree


def build_reduce_tree(reduce):
    sub_tree_lst = []
    for expr in reduce.source:
        sub_tree = build_expr_tree(expr)
        sub_tree_lst.append(sub_tree)
    res = AST.concat_subtree(sub_tree_lst)
    return res


def build_cast_tree(cast_expr):
    return build_expr_tree(cast_expr.value)


def build_select_tree(select_expr):
    tree = AST()
    ctree = build_expr_tree(select_expr.condition)
    ttree = build_expr_tree(select_expr.true_value)
    ftree = build_expr_tree(select_expr.false_value)
    bch_node = BchNode(ctree.root, ttree.root, ftree.root)
    tree.set_root(bch_node)
    return tree


def build_call_tree(call_expr):
    tree = AST()
    indice = []
    if isinstance(call_expr.func, tvm.tensor.PlaceholderOp):
        shape = call_expr.func.shape
    elif isinstance(call_expr.func, tvm.tensor.ComputeOp):
        shape = call_expr.func.output(0).shape
    else:
        raise ValueError("No implement in tvm for operation of type {}".format(type(call_expr.func)))
    for i, expr in enumerate(call_expr.args):
        expr = tvm.ir_pass.Simplify(expr)
        sub_tree = build_expr_tree(expr)
        indice.append(sub_tree.root)
    vst_node = VstNode(call_expr.func, indice, to_tuple(shape))
    tree.set_root(vst_node)
    return tree


def build_expr_tree(expr):
    p = tvm.expr
    next_steps = {
        p.Var: build_var_tree,
        p.IntImm: build_const_tree,
        p.UIntImm: build_const_tree,
        p.FloatImm: build_const_tree,
        p.Add: build_add_tree,
        p.Sub: build_sub_tree,
        p.Mul: build_mul_tree,
        p.Div: build_div_tree,
        p.EQ: build_eq_tree,
        p.NE: build_ne_tree,
        p.LE: build_le_tree,
        p.LT: build_lt_tree,
        p.GE: build_ge_tree,
        p.GT: build_gt_tree,
        p.And: build_and_tree,
        p.Or: build_or_tree,
        p.Not: build_not_tree,
        p.Reduce: build_reduce_tree,
        p.Cast: build_cast_tree,
        p.Select: build_select_tree,
        p.Call: build_call_tree,
    }
    next_step = next_steps[type(expr)]
    return next_step(expr)


def build_statement_tree(op, vst_tree, pos, reduce=False):
    tree = AST()
    body = op.body[pos]
    expr_tree = build_expr_tree(body)
    if reduce:
        inc_node = IncNode(vst_tree.root, expr_tree.root)
        tree.set_root(inc_node)
    else:
        eql_node = EqlNode(vst_tree.root, expr_tree.root)
        tree.set_root(eql_node)
    return tree
