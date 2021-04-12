import tvm
import numpy as np
import torch
import torchvision
from torchvision import transforms
from flextensor.utils import to_tuple, assert_print


def pytorch_result(img_np, label_np, params_np):
  img_torch = torch.tensor(img_np, requires_grad=False)
  label_torch = torch.tensor(label_np, requires_grad=False)
  params_torch = [torch.tensor(x, requires_grad=True) for x in params_np]
  weights_torch = []
  C1 = torch.nn.Conv2d(1, 6, 5, bias=False, padding=2)
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

  act1 = torch.tanh

  act2 = torch.relu

  t1 = C1(img_torch)
  t2 = act2(t1)
  t3 = torch.nn.functional.avg_pool2d(t2, 2)

  t4 = C2(t3)
  t5 = act2(t4)
  t6 = torch.nn.functional.avg_pool2d(t5, 2)

  t7 = C3(t6)
  t8 = act2(t7)

  t9 = t8.squeeze()

  t10 = L1(t9)
  t11 = act2(L2(t10))

  t12 = torch.nn.functional.softmax(torch.log(t11+1e-5), dim=1)

  ret = torch.nn.functional.mse_loss(t12, label_torch)

  ret.backward()

  grads = [x.grad for x in weights_torch]

  return ret, grads


def gemm(A, B, transposeA=False, transposeB=False):
    """Matrix multiplies matrix

    Args:
    -----------------------------
    A: tvm.te.tensor.Tensor
        shape [height, width]
    B: tvm.te.tensor.Tensor
        shape [width, length]
    transposeA: (optional:False) bool
    transposeB: (optional:False) bool
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [height, length]
    -----------------------------
    """
    if transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[0].value == B.shape[1].value)
        return tvm.te.compute((A.shape[1], B.shape[0]), lambda i, j: tvm.te.sum(A[k, i] * B[j, k], axis=k), requires_grad=True)
    elif transposeA and not transposeB:
        k = tvm.te.reduce_axis((0, B.shape[0]))
        assert_print(A.shape[0].value == B.shape[0].value)
        return tvm.te.compute((A.shape[1], B.shape[1]), lambda i, j: tvm.te.sum(A[k, i] * B[k, j], axis=k), requires_grad=True)
    elif not transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[1].value == B.shape[1].value)
        return tvm.te.compute((A.shape[0], B.shape[0]), lambda i, j: tvm.te.sum(A[i, k] * B[j, k], axis=k), requires_grad=True)
    else:
        k = tvm.te.reduce_axis((0, B.shape[0]))
        assert_print(A.shape[1].value == B.shape[0].value)
        return tvm.te.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=k), requires_grad=True)


def zero_pad2d(inputs, padding=0):
    """Zero padding for 2d tensor

    Args:
    -----------------------------
    inputs : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert_print(isinstance(padding, tuple), "type(padding)={}".format(type(padding)))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert_print(len(padding) == 4)

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.te.if_then_else(
                            tvm.te.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                            inputs[b, c, h - padding[0], w - padding[2]],
                            padding_zero
                            ),
        name='Padding', requires_grad=True
        )


def conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : tvm.te.tensor.Tensor
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) tvm.te.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert_print((channel_per_group * groups).value == in_channel.value)
    out_channel_per_group = out_channel // groups
    assert_print((out_channel_per_group * groups).value == out_channel.value)

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")

    padded = zero_pad2d(inputs, padding=padding)
    output = tvm.te.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda b, c, h, w: tvm.te.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
            * weight[c, rc, rh, rw]),
            axis=[rc, rw, rh]
        ), requires_grad=True
    )
    if bias is not None:
        output = tvm.te.compute(
            (batch_size, out_channel, out_h, out_w),
            lambda b, c, h, w: output[b, c, h, w] + bias[c]
        )
    return output


def ReLU(inputs):
  zeros = tvm.tir.expr.const(0, inputs.dtype)
  func = lambda *args: tvm.tir.if_then_else(inputs[args] > zeros, inputs[args], zeros)
  return tvm.te.compute(inputs.shape, func, "ReLU", requires_grad=True)


def tanh(inputs):
  func = lambda *args: tvm.tir.tanh(inputs[args])
  return tvm.te.compute(inputs.shape, func, "tanh", requires_grad=True)


def avgpool(inputs, R=2, S=2):
  r = tvm.te.reduce_axis([0, R], name="r")
  s = tvm.te.reduce_axis([0, S], name="s")

  assert len(inputs.shape) >= 2
  P = inputs.shape[-2] // R
  Q = inputs.shape[-1] // S

  C = tvm.te.compute([inputs.shape[0], inputs.shape[1], P, Q],
    lambda n, c, i, j: tvm.te.sum(inputs[n, c, i * R + r, j * S + s]/(R*S), axis=[r, s]), name="AvgPool", requires_grad=True)
  return C


def flatten_gemm(inputs, weight):
  k = tvm.te.reduce_axis([0, weight.shape[0]], name="k")
  r = tvm.te.reduce_axis([0, 1], name="r")
  s = tvm.te.reduce_axis([0, 1], name="s")
  return tvm.te.compute([inputs.shape[0], weight.shape[1]],
    lambda i, j: tvm.te.sum(inputs[i, k, r, s] * weight[k, j], axis=[k, r, s]), name="Flaten_GEMM", requires_grad=True)


def softmax_log(inputs):
  N = inputs.shape[0]
  K = inputs.shape[1]
  k = tvm.te.reduce_axis([0, K], name="k")
  k1 = tvm.te.reduce_axis([0, K], name="k1")
  max_val = tvm.te.compute([N, K], lambda n, h: tvm.te.max(inputs[n, k1], axis=[k1]), name="max_val")
  exp_val = tvm.te.compute([N, K], lambda n, h: (inputs[n, h]-max_val(n, h)), name="Softmax_exp", requires_grad=True)
  sum_val = tvm.te.compute([N], lambda n: tvm.te.sum(exp_val[n, k], axis=[k]), name="Softmax_sum", requires_grad=True)
  epsilon = tvm.tir.expr.const(1e-5, inputs.dtype) if "float" in inputs.dtype else 1
  return tvm.te.compute([N, K], lambda n, h: exp_val[n, h]/(sum_val[n] + epsilon), name="Softmax_div", requires_grad=True)


def sum_all(inputs):
  n = tvm.te.reduce_axis([0, inputs.shape[0]], name="n")
  k = tvm.te.reduce_axis([0, inputs.shape[1]], name="k")
  return tvm.te.compute([1], lambda i: tvm.te.sum(inputs[i + n, k], axis=[n, k]), name="sum_all", requires_grad=True)


def mse_loss(inputs, targets):
  N = inputs.shape[0]
  K = inputs.shape[1]
  n = tvm.te.reduce_axis([0, inputs.shape[0]], name="n")
  k = tvm.te.reduce_axis([0, inputs.shape[1]], name="k")
  # return tvm.te.compute([1], lambda i: tvm.te.sum(
  #   (inputs[i + n, k]-targets[n, k])*(inputs[i + n, k]-targets[n, k])/K, axis=[n, k]), name="mse", requires_grad=True)
  return tvm.te.compute([1], lambda i: tvm.te.sum(
    tvm.tir.power((inputs[i + n, k]-targets[i + n, k]), 2), axis=[n, k]), name="mse", requires_grad=True)


def main():
  batch = 8
  dtype = "float64"
  img = tvm.te.placeholder([batch, 1, 28, 28], dtype=dtype, name="img")
  label = tvm.te.placeholder([batch, 10], dtype=dtype, name="label")
  weight_1 = tvm.te.placeholder([6, 1, 5, 5], dtype=dtype, name="w1", requires_grad=True)
  weight_2 = tvm.te.placeholder([16, 6, 5, 5], dtype=dtype, name="w2", requires_grad=True)
  weight_3 = tvm.te.placeholder([120, 16, 5, 5], dtype=dtype, name="w3", requires_grad=True)
  weight_4 = tvm.te.placeholder([120, 84], dtype=dtype, name="w4", requires_grad=True)
  weight_5 = tvm.te.placeholder([84, 10], dtype=dtype, name="w5", requires_grad=True)

  act1 = tanh
  act2 = ReLU

  t1 = conv2d_nchw(img, weight_1, None, 1, 2, 1, 1)
  t2 = act2(t1)
  t3 = avgpool(t2)

  t4 = conv2d_nchw(t3, weight_2, None, 1, 0, 1, 1)
  t5 = act2(t4)
  t6 = avgpool(t5)

  t7 = conv2d_nchw(t6, weight_3, None, 1, 0, 1, 1)
  t8 = act2(t7)
  # t9 = avgpool(t8)

  t10 = flatten_gemm(t8, weight_4)

  t11 = act2(gemm(t10, weight_5))

  t12 = softmax_log(t11)

  # t13 = sum_all(t12)
  t13 = mse_loss(t12, label)

  d1, d2, d3, d4, d5 = tvm.te.mygradient(t13, [weight_1, weight_2, weight_3, weight_4, weight_5])

  print("Build model...")

  s = tvm.te.create_schedule([t13.op, d1.op, d2.op, d3.op, d4.op, d5.op])

  # print(tvm.lower(s, [img, label, weight_1, weight_2, weight_3, weight_4, weight_5, t12, t13, d1, d2, d3, d4, d5], simple_mode=True))

  func = tvm.build(s, [img, label, weight_1, weight_2, weight_3, weight_4, weight_5, t12, t13, d1, d2, d3, d4, d5], target="llvm")

  print("Check correctness...")

  free_vars = [weight_1, weight_2, weight_3, weight_4, weight_5]
  gradients = [d1, d2, d3, d4, d5]
  params = []
  for var in free_vars:
    shape = to_tuple(var.shape)
    var_np = np.random.uniform(-2, 2, shape).astype(dtype)
    params.append(var_np)
  img_np = np.random.uniform(0.9999, 1, to_tuple(img.shape)).astype(dtype)
  label_np = np.random.uniform(0, 1, to_tuple(label.shape)).astype(dtype)
  logit_np = np.zeros(to_tuple(t12.shape)).astype(dtype)
  ret_np = np.zeros(to_tuple(t13.shape)).astype(dtype)
  inits = []
  for var in gradients:
    shape = to_tuple(var.shape)
    var_np = np.zeros(shape).astype(dtype)
    inits.append(var_np)

  ctx = tvm.device("llvm")

  img_tvm = tvm.nd.array(img_np, ctx)
  label_tvm = tvm.nd.array(label_np, ctx)
  logit_tvm = tvm.nd.array(logit_np, ctx)
  ret_tvm = tvm.nd.array(ret_np, ctx)
  free_vars_tvm = [tvm.nd.array(x, ctx) for x in params]
  gradients_tvm = [tvm.nd.array(x, ctx) for x in inits]

  func(img_tvm, label_tvm, *free_vars_tvm, logit_tvm, ret_tvm, *gradients_tvm)

  ret_torch, grad_torch = pytorch_result(img_np, label_np, params)

  # print(ret_tvm)
  # print(ret_torch)

  # tvm.testing.assert_allclose(ret_tvm.asnumpy(), ret_torch.detach().numpy(), atol=1e-3, rtol=1e-5)
  # for i in range(len(gradients_tvm)):
  #   if i > 2:
  #     tvm.testing.assert_allclose(gradients_tvm[i].asnumpy(), grad_torch[i].detach().T.numpy(), atol=1e-3, rtol=1e-5)
  #   else:
  #     tvm.testing.assert_allclose(gradients_tvm[i].asnumpy(), grad_torch[i].detach().numpy(), atol=1e-3, rtol=1e-5)

  # print("Compare to Pytorch success!")

  print("Start training...")

  # this will be updated during training
  model_weights = []
  for var in free_vars:
    shape = to_tuple(var.shape)
    var_np = np.random.uniform(0, 0.25, shape).astype(dtype)
    model_weights.append(var_np)

  epoch = 100
  lr = 0.1

  train_set = torchvision.datasets.MNIST(".", train=True, transform=transforms.Compose([transforms.ToTensor()]))
  test_set = torchvision.datasets.MNIST(".", train=False, transform=transforms.Compose([transforms.ToTensor()]))
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True)

  for ep in range(epoch):
    for i, data in enumerate(train_loader):
      img_tvm = tvm.nd.array(data[0].numpy().astype(dtype), ctx)
      label_torch = torch.tensor(np.zeros([batch, 10]).astype(dtype))
      label_torch.scatter_(1, data[1].unsqueeze(0).T, 1.0)
      label_tvm = tvm.nd.array(label_torch.numpy(), ctx)
      weights_iter = []
      for var in model_weights:
        var_tvm = tvm.nd.array(var)
        weights_iter.append(var_tvm)
      gradients_iter = []
      for var in gradients:
        shape = to_tuple(var.shape)
        var_tvm = tvm.nd.array(np.zeros(shape).astype(dtype))
        gradients_iter.append(var_tvm)
      # print("Running...")
      func(img_tvm, label_tvm, *weights_iter, logit_tvm, ret_tvm, *gradients_iter)

      if (i) % 100 == 0:
        print("epoch=", ep+1, "iteration=", i+1, "loss=", ret_tvm.asnumpy())
        # print("logit=", logit_tvm.asnumpy())
        # print("weights")
        # print(model_weights[0])
        # print("gradients")
        # print(gradients_iter[0])

      # print("Updating...")
      for k, gradient in enumerate(gradients_iter):
        model_weights[k] -= lr * gradient.asnumpy()

      # if i > 1000:
      #   return
      


if __name__ == "__main__":
  main()