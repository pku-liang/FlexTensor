import tvm
import numpy as np
import torch
import torchvision
from torchvision import transforms
from flextensor.utils import to_tuple, assert_print


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


def flatten(inputs):
  N = inputs.shape[0]
  C = inputs.shape[1]
  H = inputs.shape[2]
  W = inputs.shape[3]
  return tvm.te.compute([N, C*H*W], lambda i, j: inputs[i, j//(H*W), j%(H*W)//W, j%W], requires_grad=True)


def softmax(inputs):
  N = inputs.shape[0]
  K = inputs.shape[1]
  k = tvm.te.reduce_axis([0, K], name="k")
  k1 = tvm.te.reduce_axis([0, K], name="k1")
  max_val = tvm.te.compute([N, K], lambda n, h: tvm.te.max(inputs[n, k1], axis=[k1]), name="max_val")
  exp_val = tvm.te.compute([N, K], lambda n, h: tvm.tir.exp(inputs[n, h]-max_val(n, h)), name="Softmax_exp", requires_grad=True)
  sum_val = tvm.te.compute([N], lambda n: tvm.te.sum(exp_val[n, k], axis=[k]), name="Softmax_sum", requires_grad=True)
  epsilon = tvm.tir.expr.const(1e-5, inputs.dtype) if "float" in inputs.dtype else 1
  return tvm.te.compute([N, K], lambda n, h: exp_val[n, h]/(sum_val[n] + epsilon), name="Softmax_div", requires_grad=True)


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
  batch = 32
  dtype = "float64"
  img = tvm.te.placeholder([batch, 1, 28, 28], dtype=dtype, name="img")
  label = tvm.te.placeholder([batch, 10], dtype=dtype, name="label")
  weight_1 = tvm.te.placeholder([28*28, 10], dtype=dtype, name="w1", requires_grad=True)

  t1 = flatten(img)

  t2 = gemm(t1, weight_1)

  t3 = softmax(t2)

  t4 = mse_loss(t3, label)

  d1, = tvm.te.mygradient(t4, [weight_1])

  print("Build model...")

  s = tvm.te.create_schedule([t4.op, d1.op])

  func = tvm.build(s, [img, label, weight_1, t4, d1], target="llvm")

  print("Check correctness...")

  free_vars = [weight_1]
  gradients = [d1]
  params = []
  for var in free_vars:
    shape = to_tuple(var.shape)
    var_np = np.random.uniform(-2, 2, shape).astype(dtype)
    params.append(var_np)
  img_np = np.random.uniform(0.9999, 1, to_tuple(img.shape)).astype(dtype)
  label_np = np.random.uniform(0, 1, to_tuple(label.shape)).astype(dtype)
  ret_np = np.zeros(to_tuple(t4.shape)).astype(dtype)
  inits = []
  for var in gradients:
    shape = to_tuple(var.shape)
    var_np = np.zeros(shape).astype(dtype)
    inits.append(var_np)

  ctx = tvm.device("llvm")

  img_tvm = tvm.nd.array(img_np, ctx)
  label_tvm = tvm.nd.array(label_np, ctx)
  ret_tvm = tvm.nd.array(ret_np, ctx)
  free_vars_tvm = [tvm.nd.array(x, ctx) for x in params]
  gradients_tvm = [tvm.nd.array(x, ctx) for x in inits]

  func(img_tvm, label_tvm, *free_vars_tvm, ret_tvm, *gradients_tvm)

  print("Start training...")

  # this will be updated during training
  model_weights = []
  for var in free_vars:
    shape = to_tuple(var.shape)
    var_np = np.random.uniform(-0.5, 0.5, shape).astype(dtype)
    model_weights.append(var_np)

  epoch = 100
  lr = 0.0002

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
      func(img_tvm, label_tvm, *weights_iter, ret_tvm, *gradients_iter)

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