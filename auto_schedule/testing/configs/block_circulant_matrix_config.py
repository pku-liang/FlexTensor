block_circulant_matrix_shapes = []

for shape in [(1024, 256), (1024, 512), (1024, 40)]:
    for factor in [8, 16]:
        block_circulant_matrix_shapes.append((*shape, factor))
