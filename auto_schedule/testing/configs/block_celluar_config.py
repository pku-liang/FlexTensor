def power2_list(bounds):
    lower_bound, upper_bound = 1, 1
    if isinstance(bounds, int):
        upper_bound = bounds
    elif isinstance(bounds, (list, tuple)):
        lower_bound, upper_bound = bounds
    else:
        raise ValueError

    ret = []
    cur = lower_bound
    while cur <= upper_bound:
        ret.append(cur)
        cur *= 2
    return ret


block_celluar_shapes = []

for N in power2_list((32, 256)):
    block_celluar_shapes.append((N, N))
