old_gemm_shapes = [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048)
]

gemm_shapes = []
for i in range(5, 13):
    for j in range(5, 13):
        for k in range(5, 13):
            gemm_shapes.append([2**i, 2**k, 2**j])


test_gemm_shapes = [
    # batch
    # height
    # width
    # length

    # open
    (1, 1024, 1024, 1024),
    (2, 512, 512, 512),
    (3, 1024, 32, 1024),
    # confidential
    (16, 4096, 128, 1024),
    (32, 28, 1024, 28),
]