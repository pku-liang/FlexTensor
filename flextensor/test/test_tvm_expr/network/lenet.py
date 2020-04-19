import tvm
from flextensor.nn import conv2d_nchw, gemm


def ReLU(inputs):
  func = lambda *args: tvm.tir.if_then_else(inputs[args] > 0.0, inputs[args], 0.0)
  return tvm.te.compute(inputs.shape, func, "ReLU")


def avgpool(inputs, R=2, S=2):
  r = tvm.te.reduce_axis([0, R], name="r")
  s = tvm.te.reduce_axis([0, S], name="s")

  assert len(inputs.shape) >= 2
  P = inputs.shape[-2] // R
  Q = inputs.shape[-1] // S

  C = tvm.te.compute([inputs.shape[0], inputs.shape[1], P, Q],
    lambda n, c, i, j: tvm.te.sum(inputs[n, c, i * R + r, j * S + s]/(R*S), axis=[r, s]), name="AvgPool")
  return C


def flatten_gemm(inputs, weight):
  k = tvm.te.reduce_axis([0, weight.shape[0]], name="k")
  r = tvm.te.reduce_axis([0, 1], name="r")
  s = tvm.te.reduce_axis([0, 1], name="s")
  return tvm.te.compute([inputs.shape[0], weight.shape[1]],
    lambda i, j: tvm.te.sum(inputs[i, k, r, s] * weight[k, j], axis=[k, r, s]), name="Flaten_GEMM")


def softmax(inputs):
  K = inputs.shape[-1]
  k = tvm.te.reduce_axis([0, K], name="k")
  exp_val = tvm.te.compute(inputs.shape, lambda *args: tvm.te.exp(inputs[args]), name="Softmax_exp")
  sum_val = tvm.te.compute(exp_val.shape[:-1], lambda *args: tvm.te.sum(exp_val[tuple(list(args) + [k])], axis=[k]), name="Softmax_sum")
  return tvm.te.compute(exp_val.shape, lambda *args: exp_val[args]/sum_val[(args[:-1])], name="Softmax_div")


def main():
  batch = 2
  dtype = "float32"
  img = tvm.te.placeholder([batch, 1, 32, 32], dtype=dtype, name="img")
  weight_1 = tvm.te.placeholder([6, 1, 5, 5], dtype=dtype, name="w1")
  weight_2 = tvm.te.placeholder([16, 6, 5, 5], dtype=dtype, name="w2")
  weight_3 = tvm.te.placeholder([120, 16, 5, 5], dtype=dtype, name="w3")
  weight_4 = tvm.te.placeholder([120, 84], dtype=dtype, name="w4")
  weight_5 = tvm.te.placeholder([84, 10], dtype=dtype, name="w5")

  t1 = conv2d_nchw(img, weight_1, None, 1, 0, 1, 1)
  t2 = ReLU(t1)
  t3 = avgpool(t2)

  t4 = conv2d_nchw(t3, weight_2, None, 1, 0, 1, 1)
  t5 = ReLU(t4)
  t6 = avgpool(t5)

  t7 = conv2d_nchw(t6, weight_3, None, 1, 0, 1, 1)
  t8 = ReLU(t7)
  t9 = avgpool(t8)

  t10 = flatten_gemm(t9, weight_4)

  t11 = gemm(t10, weight_5)

  t12 = softmax(t11)

  s = tvm.te.create_schedule(t12.op)

  print(tvm.lower(s, [img, weight_1, weight_2, weight_3, weight_4, weight_5], simple_mode=True))

  tvm.te.mygradient(t12, [weight_1])


if __name__ == "__main__":
  main()