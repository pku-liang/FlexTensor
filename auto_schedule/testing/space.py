import tvm
import numpy as np
from auto_schedule.utils import assert_print, gen_enum, any_factor_split, get_factor_lst


def able_inline(op, down_graph):
    is_compute = isinstance(op, tvm.tensor.ComputeOp)
    has_reduce = hasattr(op, "reduce_axis") and op.reduce_axis
    is_output = False
    for i in range(op.num_outputs):
        if op.output(i) not in down_graph:
            is_output = True
            break
    return is_compute and (not has_reduce) and (not is_output)


# class SubSpace(object):
#     def __init__(self, entities):
#         assert_print(isinstance(entities, (list, tuple)) and len(entities) > 0)
#         self.entities = entities
#         self.begin = 0
#         self.end = len(self.entities)

#     def get_entity(self, p):
#         if len(self.entities) < 1:
#             raise RuntimeError("Query from empty space")
#         if 0 <= p < self.end:
#             return self.entities[p]
#         else:
#             raise RuntimeError("Space pointer out of range")

#     def range(self, p, left, right=None):
#         if right is None:
#             right = left
#         left = p - left if p - left >= 0 else 0
#         right = p + right if p + right <= self.end else self.end
#         return range(left, right), self.entities[left:right]

#     def __len__(self):
#         return self.end


class Space(object):
    def __init__(self):
        self.subspaces = {}
        self.types = {}
        self.valid_type_keys = ["spatial", "reduce", "inline", "unroll"]
        for type_key in self.valid_type_keys:
            self.types[type_key] = []
        self.dim = 0

    def add_subspace(self, name, subspace, type_key, override=False):
        if name in self.subspaces and not override:
            raise RuntimeError("Same subspace name")
        assert_print(type_key in self.valid_type_keys)
        self.subspaces[name] = subspace
        self.types[type_key].append(name)
        self.dim += subspace.dim

    def items(self):
        return self.subspaces.items()

    def __len__(self):
        ret = 1
        for _, subspace in self.subspaces.items():
            ret *= len(subspace)
        return ret

    def length(self):
        ret = {}
        total = 1
        added = 0
        for name, subspace in self.subspaces.items():
            ret[name] = len(subspace)
            total *= ret[name]
            added += ret[name]
        ret["total"] = total
        ret["added"] = added
        return ret


DirectedSubSpaceTypeKeys = ["split"]
UndirectedSubSpaceTypeKeys = ["unroll", "inline"]


class SubSpace(object):
    def __init__(self):
        self.dim = 0
        self.static_entities = []
        self.size = 0
        self.num_direction = 0

    def random_entity(self):
        return np.random.choice(self.static_entities)

    def next_entity(self, *args, **kwargs):
        raise NotImplementedError()

    def get_entity(self, p):
        return self.static_entities[p]

    def get_direction(self, num):
        raise NotImplementedError()

    def __len__(self):
        return self.size


class SplitSpace(SubSpace):
    def __init__(self, dim, total):
        super(SplitSpace, self).__init__()
        self.total = total
        self.dim = dim
        self.static_entities = any_factor_split(total, dim)
        self.size = len(self.static_entities)
        self.num_direction = dim * (dim - 1)
        self.directions = []
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    self.directions.append((i, j))
        self.type_key = "split"
    
    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return self.static_entities[pos]
        elif len(d) == 2:
            asc_pos, dec_pos = d[0], d[1]
            assert_print(0 <= asc_pos < self.dim)
            assert_print(0 <= dec_pos < self.dim)
            assert_print(asc_pos != dec_pos)
            current = self.static_entities[pos]
            ret = current.copy()
            left = current[asc_pos] * current[dec_pos]
            tmp = current[asc_pos] + 1
            while tmp <= left:
                if left % tmp == 0:
                    break
                tmp += 1
            tmp = min(tmp, left)
            ret[asc_pos] = tmp
            ret[dec_pos] = left // tmp
            return self.static_entities.index(ret)
        else:
            raise RuntimeError(
                "Not support for direction more than two dims: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class UnrollSpace(SubSpace):
    def __init__(self, steps):
        super(UnrollSpace, self).__init__()
        self.dim = 2
        self.static_entities = []
        self.steps = steps
        for step in steps:
            for explicit in [0, 1]:
                self.static_entities.append([step, explicit])
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "unroll"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class InlineSpace(SubSpace):
    def __init__(self, inline_op_pos, op_num):
        self.dim = op_num
        num_inline_ops = len(inline_op_pos)
        enums = gen_enum([1, 0], num_inline_ops)
        self.static_entities = []
        for enum in enums:
            entity = [0] * op_num
            for i in range(num_inline_ops):
                entity[inline_op_pos[i]] = enum[i]
            self.static_entities.append(entity)
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "inline"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


def generate_inline_space(op_lst, down_graph):
    inline_op_pos = []
    for i, op in enumerate(op_lst):
        if able_inline(op, down_graph):
            inline_op_pos.append(i)
    return InlineSpace(inline_op_pos, len(op_lst))


def generate_split_space(extent, nparts):
    return SplitSpace(nparts, extent)


def generate_unroll_space():
    return UnrollSpace([0, 1, 512, 1500])


def generate_space_intra_op(op, down_graph, slevel=4, rlevel=3):
    spatial_axis_names = [x.var.name for x in op.axis]
    spatial_axis_extents = [x.dom.extent.value for x in op.axis]
    reduced_axis_names = [x.var.name for x in op.reduce_axis]
    reduced_axis_extents = [x.dom.extent.value for x in op.reduce_axis]

    ##############################################################
    # generate space: 
    schedule_space = Space()
    # - split space
    for i, (name, extent) in enumerate(zip(spatial_axis_names, spatial_axis_extents)):
        split_space = generate_split_space(extent, slevel)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "spatial")
    for i, (name, extent) in enumerate(zip(reduced_axis_names, reduced_axis_extents)):
        split_space = generate_split_space(extent, rlevel)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "reduce")
    # -unroll space
    unroll_space = generate_unroll_space()
    schedule_space.add_subspace("unroll", unroll_space, "unroll")
    # - other spaces can be added

    return schedule_space


def generate_space_inter_op(op_lst, down_graph):

    ##############################################################
    # generate space:
    schedule_space = Space()
    # - inline space
    inline_space = generate_inline_space(op_lst, down_graph)
    schedule_space.add_subspace("inline", inline_space, "inline")
    # - other spaces can be added

    return schedule_space