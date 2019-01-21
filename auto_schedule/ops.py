import tvm


def gemv(A, V):
    k = tvm.reduce_axis((0, A.shape[1]), name="k")
    C = tvm.compute((A.shape[0],), lambda i: tvm.sum(A[i, k] * V[k], axis=k), name="C")
    return C


def matmul(A, B):
    k = tvm.reduce_axis((0, A.shape[1]), name="k")
    C = tvm.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name="C")
    return C


def conv2d(A, W, stride=1, padding=0):
    N_out = max(0, int(A.shape[0] + padding * 2 - W.shape[0]) // stride) + 1
    M_out = max(0, int(A.shape[1] + padding * 2 - W.shape[1]) // stride) + 1
    Apad = tvm.compute((A.shape[0] + 2 * padding, A.shape[1] + 2 * padding),
                       lambda i, j: tvm.select(tvm.all(i >= padding, j >= padding, i < A.shape[0] + padding, j < A.shape[1] + padding),
                                               A[i - padding, j - padding], 0.0), name="Apad")
    rx, ry = tvm.reduce_axis((0, W.shape[0]), name="rx"), tvm.reduce_axis((0, W.shape[1]), name="ry")
    Output = tvm.compute((N_out, M_out),
                         lambda i, j: tvm.sum(Apad[i * stride + rx, j * stride + ry] * W[rx, ry], axis=[rx, ry]),
                         name="Output")
    return Output


def conv2d_batch(A, W, stride=1, padding=0):
    N_out = max(0, int(A.shape[1] + padding * 2 - W.shape[0]) // stride) + 1
    M_out = max(0, int(A.shape[2] + padding * 2 - W.shape[1]) // stride) + 1
    Apad = tvm.compute((A.shape[0], A.shape[1] + 2 * padding, A.shape[2] + 2 * padding),
                       lambda b, i, j: tvm.select(tvm.all(i >= padding, j >= padding, i < A.shape[1] + padding, j < A.shape[2] + padding),
                                                  A[b, i - padding, j - padding], 0.0), name="Apad")
    rx, ry = tvm.reduce_axis((0, W.shape[0]), name="rx"), tvm.reduce_axis((0, W.shape[1]), name="ry")
    Output = tvm.compute((A.shape[0], N_out, M_out),
                         lambda b, i, j: tvm.sum(Apad[b, i * stride + rx, j * stride + ry] * W[rx, ry], axis=[rx, ry]),
                         name="Output")
    return Output


def conv2d_channel(A, W, stride=1, padding=0):
    N_out = max(0, int(A.shape[0] + padding * 2 - W.shape[0]) // stride) + 1
    M_out = max(0, int(A.shape[1] + padding * 2 - W.shape[1]) // stride) + 1
    Apad = tvm.compute((A.shape[0] + 2 * padding, A.shape[1] + 2 * padding, A.shape[2]),
                       lambda i, j, k: tvm.select(tvm.all(i >= padding, j >= padding, i < A.shape[0] + padding, j < A.shape[1] + padding),
                                               A[i - padding, j - padding, k], 0.0), name="Apad")
    rx, ry = tvm.reduce_axis((0, W.shape[0]), name="rx"), tvm.reduce_axis((0, W.shape[1]), name="ry")
    rc = tvm.reduce_axis((0, A.shape[2]), name="rc")
    Output = tvm.compute((N_out, M_out, W.shape[3]),
                         lambda i, j, k: tvm.sum(Apad[i * stride + rx, j * stride + ry, rc] * W[rx, ry, rc, k],
                                                 axis=[rx, ry, rc]),
                         name="Output")
    return Output


def conv2d_channel_batch(A, W, stride=1, padding=0):
    N_out = max(0, int(A.shape[1] + padding * 2 - W.shape[0]) // stride) + 1
    M_out = max(0, int(A.shape[2] + padding * 2 - W.shape[1]) // stride) + 1
    Apad = tvm.compute((A.shape[0], A.shape[1] + 2 * padding, A.shape[2] + 2 * padding, A.shape[3]),
                       lambda b, i, j, k: tvm.select(tvm.all(i >= padding, j >= padding, i < A.shape[1] + padding, j < A.shape[2] + padding),
                                                  A[b, i - padding, j - padding, k], 0.0), name="Apad")
    rx, ry = tvm.reduce_axis((0, W.shape[0]), name="rx"), tvm.reduce_axis((0, W.shape[1]), name="ry")
    rc = tvm.reduce_axis((0, A.shape[3]), name="rc")
    Output = tvm.compute((A.shape[0], N_out, M_out, W.shape[3]),
                         lambda b, i, j, k: tvm.sum(Apad[b, i * stride + rx, j * stride + ry, rc] * W[rx, ry, rc, k],
                                                    axis=[rx, ry, rc]),
                         name="Output")
    return Output


def conv3d(A, W, stride=1, padding=0):
    N_out = max(0, int(A.shape[0] + padding * 2 - W.shape[0]) // stride) + 1
    M_out = max(0, int(A.shape[1] + padding * 2 - W.shape[1]) // stride) + 1
    P_out = max(0, int(A.shape[2] + padding * 2 - W.shape[2]) // stride) + 1
    Apad = tvm.compute((A.shape[0] + 2 * padding, A.shape[1] + 2 * padding, A.shape[2] + 2 * padding),
                       lambda i, j, k: tvm.select(
                           tvm.all(i >= padding, j >= padding, k >= padding, i < A.shape[0] + padding, j < A.shape[1] + padding,
                                   k < A.shape[2] + padding),
                           A[i - padding, j - padding, k - padding], 0.0), name="Apad")
    rx, ry, rz = tvm.reduce_axis((0, W.shape[0]), name="rx"), tvm.reduce_axis((0, W.shape[1]), name="ry"), tvm.reduce_axis((0, W.shape[2]),
                                                                                                         name="rz")
    Output = tvm.compute((N_out, M_out, P_out),
                         lambda i, j, k: tvm.sum(
                             Apad[i * stride + rx, j * stride + ry, k * stride + rz] * W[rx, ry, rz],
                             axis=[rx, ry, rz]),
                         name="Output")
    return Output


def conv3d_batch(A, W, stride=1, padding=0):
    N_out = max(0, int(A.shape[1] + padding * 2 - W.shape[0]) // stride) + 1
    M_out = max(0, int(A.shape[2] + padding * 2 - W.shape[1]) // stride) + 1
    P_out = max(0, int(A.shape[3] + padding * 2 - W.shape[2]) // stride) + 1
    Apad = tvm.compute((A.shape[0], A.shape[1] + 2 * padding, A.shape[2] + 2 * padding, A.shape[3] + 2 * padding),
                       lambda b, i, j, k: tvm.select(
                           tvm.all(i >= padding, j >= padding, k >= padding, i < A.shape[1] + padding, j < A.shape[2] + padding,
                                   k < A.shape[3] + padding),
                           A[b, i - padding, j - padding, k - padding], 0.0), name="Apad")
    rx, ry, rz = tvm.reduce_axis((0, W.shape[0]), name="rx"), tvm.reduce_axis((0, W.shape[1]), name="ry"), tvm.reduce_axis((0, W.shape[2]),
                                                                                                         name="rz")
    Output = tvm.compute((A.shape[0], N_out, M_out, P_out),
                         lambda b, i, j, k: tvm.sum(
                             Apad[b, i * stride + rx, j * stride + ry, k * stride + rz] * W[rx, ry, rz],
                             axis=[rx, ry, rz]),
                         name="Output")
    return Output


def conv3d_channel(A, W, stride=1, padding=0):
    N_out = max(0, int(A.shape[0] + padding * 2 - W.shape[0]) // stride) + 1
    M_out = max(0, int(A.shape[1] + padding * 2 - W.shape[1]) // stride) + 1
    P_out = max(0, int(A.shape[2] + padding * 2 - W.shape[2]) // stride) + 1
    Apad = tvm.compute((A.shape[0] + 2 * padding, A.shape[1] + 2 * padding, A.shape[2] + 2 * padding, A.shape[3]),
                       lambda i, j, k, c: tvm.select(
                           tvm.all(i >= padding, j >= padding, k >= padding, i < A.shape[0] + padding, j < A.shape[1] + padding,
                                   k < A.shape[2] + padding),
                           A[i - padding, j - padding, k - padding, c], 0.0), name="Apad")
    rx, ry, rz, rc = tvm.reduce_axis((0, W.shape[0]), name="rx"), tvm.reduce_axis((0, W.shape[1]), name="ry"), \
                     tvm.reduce_axis((0, W.shape[2]), name="rz"), tvm.reduce_axis((0, A.shape[3]), name="rc")
    Output = tvm.compute((N_out, M_out, P_out, W.shape[4]),
                         lambda i, j, k, o: tvm.sum(
                             Apad[i * stride + rx, j * stride + ry, k * stride + rz, rc] * W[rx, ry, rz, rc, o],
                             axis=[rx, ry, rz, rc]),
                         name="Output")
    return Output


def conv3d_channel_batch(A, W, stride=1, padding=0):
    N_out = max(0, int(A.shape[1] + padding * 2 - W.shape[0]) // stride) + 1
    M_out = max(0, int(A.shape[2] + padding * 2 - W.shape[1]) // stride) + 1
    P_out = max(0, int(A.shape[3] + padding * 2 - W.shape[2]) // stride) + 1
    Apad = tvm.compute((A.shape[0], A.shape[1] + 2 * padding, A.shape[2] + 2 * padding, A.shape[3] + 2 * padding, A.shape[4]),
                       lambda b, i, j, k, c: tvm.select(
                           tvm.all(i >= padding, j >= padding, k >= padding, i < A.shape[1] + padding, j < A.shape[2] + padding,
                                   k < A.shape[3] + padding),
                           A[b, i - padding, j - padding, k - padding, c], 0.0), name="Apad")
    rx, ry, rz, rc = tvm.reduce_axis((0, W.shape[0]), name="rx"), tvm.reduce_axis((0, W.shape[1]), name="ry"), \
                     tvm.reduce_axis((0, W.shape[2]), name="rz"), tvm.reduce_axis((0, A.shape[4]), name="rc")
    Output = tvm.compute((A.shape[0], N_out, M_out, P_out, W.shape[4]),
                         lambda b, i, j, k, o: tvm.sum(
                             Apad[b, i * stride + rx, j * stride + ry, k * stride + rz, rc] * W[rx, ry, rz, rc, o],
                             axis=[rx, ry, rz, rc]),
                         name="Output")
    return Output
