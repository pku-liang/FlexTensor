import tvm
from auto_schedule.utils import assert_print, gen_enum, any_factor_split


def able_inline(op, down_graph):
    is_compute = isinstance(op, tvm.tensor.ComputeOp)
    has_reduce = hasattr(op, "reduce_axis") and op.reduce_axis
    is_output = False
    for i in range(op.num_outputs):
        if op.output(i) not in down_graph:
            is_output = True
            break
    return is_compute and (not has_reduce) and (not is_output)


class SubSpace(object):
    def __init__(self, entities):
        assert_print(isinstance(entities, (list, tuple)) and len(entities) > 0)
        self.entities = entities
        self.begin = 0
        self.end = len(self.entities)

    def get_entity(self, p):
        if len(self.entities) < 1:
            raise RuntimeError("Query from empty space")
        if 0 <= p < self.end:
            return self.entities[p]
        else:
            raise RuntimeError("Space pointer out of range")

    def range(self, p, left, right=None):
        if right is None:
            right = left
        left = p - left if p - left >= 0 else 0
        right = p + right if p + right <= self.end else self.end
        return range(left, right), self.entities[left:right]

    def __len__(self):
        return self.end


class Space(object):
    def __init__(self):
        self.subspaces = {}
        self.types = {}
        self.valid_type_keys = ["spatial", "reduce", "inline", "unroll"]
        for type_key in self.valid_type_keys:
            self.types[type_key] = []

    def add_subspace(self, name, subspace, type_key, override=False):
        if name in self.subspaces and not override:
            raise RuntimeError("Same subspace name")
        assert_print(type_key in self.valid_type_keys)
        self.subspaces[name] = subspace
        self.types[type_key].append(name)

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


def generate_inline_space(op_lst, down_graph):
    inline_op_pos = []
    for i, op in enumerate(op_lst):
        if able_inline(op, down_graph):
            inline_op_pos.append(i)
    length = len(inline_op_pos)
    enums = gen_enum([True, False], length)
    entities = []
    for enum in enums:
        entity = [inline_op_pos[i] if enum[i] else -1 for i in range(length)]
        entities.append(entity)
    return SubSpace(entities)


def generate_split_space(extent, nparts):
    entities = any_factor_split(extent, nparts)
    return SubSpace(entities)


def generate_unroll_space():
    entities = [[0, 0], [0, 1], 
                [512, 0], [512, 1],
                [1500, 0], [1500, 1]]
    return SubSpace(entities)


def generate_space_intra_op(op, down_graph, level=4):
    spatial_axis_names = [x.var.name for x in op.axis]
    spatial_axis_extents = [x.dom.extent.value for x in op.axis]
    reduced_axis_names = [x.var.name for x in op.reduce_axis]
    reduced_axis_extents = [x.dom.extent.value for x in op.reduce_axis]

    ##############################################################
    # generate space: 
    schedule_space = Space()
    # - split space
    for i, (name, extent) in enumerate(zip(spatial_axis_names, spatial_axis_extents)):
        split_space = generate_split_space(extent, level)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "spatial")
    for i, (name, extent) in enumerate(zip(reduced_axis_names, reduced_axis_extents)):
        split_space = generate_split_space(extent, level-1)
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