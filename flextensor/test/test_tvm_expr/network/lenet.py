import tvm
import numpy as np
import torch
from flextensor.nn import conv2d_nchw, gemm
from flextensor.utils import to_tuple


def pytorch_result(img_np, params_np):
  img_torch = torch.tensor(img_np, requires_grad=False)
  params_torch = [torch.tensor(x, requires_grad=True) for x in params_np]
  weights_torch = []
  C1 = torch.nn.Conv2d(1, 6, 5, bias=False)
  C2 = torch.nn.Conv2d(6, 16, 5, bias=False)
  C3 = torch.nn.Conv2d(16, 120, 5, bias=False)
  L1 = torch.nn.Linear(120, 84, bias=False)
  L2 = torch.nn.Linear(84, 10, bias=False)

  layers = [C1, C2, C3, L1, L2]
  for i, l in enumerate(layers):
    if (i > 2):
      l.weight = torch.nn.Parameter(params_torch[i].T)
    else:
      l.weight = torch.nn.Parameter(params_torch[i])
    weights_torch.append(l.weight)

  t1 = C1(img_torch)
  t2 = torch.relu(t1)
  t3 = torch.nn.functional.avg_pool2d(t2, 2)

  t4 = C2(t3)
  t5 = torch.relu(t4)
  t6 = torch.nn.functional.avg_pool2d(t5, 2)

  t7 = C3(t6)
  t8 = torch.relu(t7)

  t9 = t8.squeeze()

  t10 = L1(t9)
  t11 = torch.relu(L2(t10))

  ret = torch.nn.functional.softmax(torch.log(t11+1e-5), dim=1)

  ret.sum().backward()

  grads = [x.grad for x in weights_torch]

  return ret, grads


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


def softmax_log(inputs):
  N = inputs.shape[0]
  K = inputs.shape[1]
  k = tvm.te.reduce_axis([0, K], name="k")
  k1 = tvm.te.reduce_axis([0, K], name="k1")
  # mean_val = tvm.te.compute([N], lambda n: tvm.te.sum(inputs[n, k1]/K, axis=[k1]), name="mean_val")
  exp_val = tvm.te.compute([N, K], lambda n, h: (inputs[n, h]), name="Softmax_exp")
  sum_val = tvm.te.compute([N], lambda n: tvm.te.sum(exp_val[n, k], axis=[k]), name="Softmax_sum")
  return tvm.te.compute([N, K], lambda n, h: exp_val[n, h]/(sum_val[n] + 1e-5), name="Softmax_div")


def sum_all(inputs):
  n = tvm.te.reduce_axis([0, inputs.shape[0]], name="n")
  k = tvm.te.reduce_axis([0, inputs.shape[1]], name="k")
  return tvm.te.compute([1], lambda i: tvm.te.sum(inputs[i + n, k], axis=[n, k]), name="sum_all")


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
  # t9 = avgpool(t8)

  t10 = flatten_gemm(t8, weight_4)

  t11 = ReLU(gemm(t10, weight_5))

  t12 = softmax_log(t11)

  t13 = sum_all(t12)


  d1, d2, d3, d4, d5 = tvm.te.mygradient(t13, [weight_1, weight_2, weight_3, weight_4, weight_5])

  s = tvm.te.create_schedule([t13.op, d1.op, d2.op, d3.op, d4.op, d5.op])

  func = tvm.build(s, [img, weight_1, weight_2, weight_3, weight_4, weight_5, t12, d1, d2, d3, d4, d5], target="llvm")

  free_vars = [weight_1, weight_2, weight_3, weight_4, weight_5]
  gradients = [d1, d2, d3, d4, d5]
  params = []
  for var in free_vars:
    shape = to_tuple(var.shape)
    var_np = np.random.uniform(-1, 1, shape).astype(dtype)
    params.append(var_np)
  img_np = np.random.uniform(-1, 1, to_tuple(img.shape)).astype(dtype)
  ret_np = np.zeros(to_tuple(t12.shape)).astype(dtype)
  inits = []
  for var in gradients:
    shape = to_tuple(var.shape)
    var_np = np.zeros(shape).astype(dtype)
    inits.append(var_np)

  ctx = tvm.context("llvm")

  img_tvm = tvm.nd.array(img_np, ctx)
  ret_tvm = tvm.nd.array(ret_np, ctx)
  free_vars_tvm = [tvm.nd.array(x, ctx) for x in params]
  gradients_tvm = [tvm.nd.array(x, ctx) for x in inits]

  func(img_tvm, *free_vars_tvm, ret_tvm, *gradients_tvm)

  ret_torch, grad_torch = pytorch_result(img_np, params)

  print(ret_tvm)
  print(ret_torch)

  tvm.testing.assert_allclose(ret_tvm.asnumpy(), ret_torch.detach().numpy(), rtol=1e-5)
  for i in range(len(gradients_tvm)):
    if i > 2:
      tvm.testing.assert_allclose(gradients_tvm[i].asnumpy(), grad_torch[i].detach().T.numpy(), rtol=1e-5)
    else:
      tvm.testing.assert_allclose(gradients_tvm[i].asnumpy(), grad_torch[i].detach().numpy(), rtol=1e-5)



if __name__ == "__main__":
  main()