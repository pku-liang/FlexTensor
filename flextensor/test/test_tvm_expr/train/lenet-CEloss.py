import tvm
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from flextensor.utils import to_tuple, assert_print

enable_relu = False

def gemm(A, B):
  k = tvm.te.reduce_axis((0, B.shape[0]))
  assert_print(A.shape[1].value == B.shape[0].value)
  return tvm.te.compute([A.shape[0], B.shape[1]], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=k), requires_grad=True)


def zero_pad2d(inputs, padding=0):
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


def avgpool(inputs, R=2, S=2):
  r = tvm.te.reduce_axis([0, R], name="r")
  s = tvm.te.reduce_axis([0, S], name="s")

  assert len(inputs.shape) >= 2
  P = inputs.shape[-2] // R
  Q = inputs.shape[-1] // S

  C = tvm.te.compute([inputs.shape[0], inputs.shape[1], P, Q],
    lambda n, c, i, j: tvm.te.sum(inputs[n, c, i * R + r, j * S + s]/(R*S), axis=[r, s]), name="AvgPool", requires_grad=True)
  return C


def flatten(inputs):
  '''
  inputs: [batch, channel, height, width]
  return: [batch, channel * height * width]
  '''
  assert(inputs.shape[1].value*inputs.shape[2].value*inputs.shape[3].value == 400)
  return tvm.te.compute([inputs.shape[0], inputs.shape[1]*inputs.shape[2]*inputs.shape[3]], 
    lambda i, j: inputs[i, j//(inputs.shape[2]*inputs.shape[3]), (j%(inputs.shape[2]*inputs.shape[3])) // inputs.shape[3], 
      j % inputs.shape[3]], name="flatten", requires_grad=True)


def cross_entropy(inputs, targets):
  N, C = inputs.shape
  c = tvm.te.reduce_axis([0, C], "c")
  k1 = tvm.te.reduce_axis([0, C], name="k1")
  # First compute the maximum for each batch
  max_val = tvm.te.compute([N], lambda n: tvm.te.max(inputs[n, k1], axis=[k1]), name="max_val")
  # Use the log_softmax trick to avoid overflow
  sum_val = tvm.te.compute([N], lambda i: tvm.te.sum(tvm.tir.exp(inputs[i, c]-max_val[i]), axis=[c]), "sum_val")
  rrn = tvm.te.reduce_axis([0, N], "rrn")
  rrc = tvm.te.reduce_axis([0, C], "rrc")
  return tvm.te.compute([1],
    lambda i: tvm.te.sum(
      targets[i+rrn, rrc] * ((tvm.tir.log(sum_val[i+rrn])+max_val[i+rrn]) - inputs[i+rrn, rrc]*targets[i+rrn, rrc])/(N),
      axis=[rrn, rrc]),
    name="cross_entropy", requires_grad=True)

def ReLU(inputs):
  zeros = tvm.tir.expr.const(0, inputs.dtype)
  func = lambda *args: tvm.tir.if_then_else(inputs[args] > zeros, inputs[args], zeros)
  return tvm.te.compute(inputs.shape, func, "ReLU", requires_grad=True)

def main():
  '''
  conv[outchannel, inchannel, kernel_h, kernel_w], 
  simplified formula: output_h = input_h + 2*padding - kernel + 1

  Non_ReLu version:
  [batch, 1, 28, 28] -> conv[6, 1, 3, 3], padding=1 -> [batch, 6, 28, 28] -> avgpool -> [batch, 6, 14, 14]
  -> conv[16, 6, 5, 5], padding=0 -> [batch, 16, 10, 10] -> avgpool -> [batch, 16, 5, 5] -> flatten -> [batch, 400]
  -> gemm[400, 120] -> [batch, 120] -> gemm[120, 84] -> [batch, 84] -> gemm[84, 10] -> [batch, 10] -> CE_loss
  '''
  batch = 4
  dtype = "float64"
  img = tvm.te.placeholder([batch, 1, 28, 28], dtype=dtype, name="img")
  label = tvm.te.placeholder([batch, 10], dtype=dtype, name="label")
  weight_1 = tvm.te.placeholder([6, 1, 3, 3], dtype=dtype, name="w1", requires_grad=True)
  weight_2 = tvm.te.placeholder([16, 6, 5, 5], dtype=dtype, name="w2", requires_grad=True)
  weight_3 = tvm.te.placeholder([400, 120], dtype=dtype, name="w3", requires_grad=True)
  weight_4 = tvm.te.placeholder([120, 84], dtype=dtype, name="w4", requires_grad=True)
  weight_5 = tvm.te.placeholder([84, 10], dtype=dtype, name="w5", requires_grad=True)

  if enable_relu:
    t1 = conv2d_nchw(img, weight_1, None, 1, 1, 1, 1)
    t2 = ReLU(t1)
    t3 = avgpool(t2)

    t4 = conv2d_nchw(t3, weight_2, None, 1, 0, 1, 1)
    t5 = ReLU(t4)
    t6 = avgpool(t5)

    t7 = flatten(t6)
    t8 = gemm(t7, weight_3)
    t9 = gemm(t8, weight_4)
    t10 = ReLU(t9)

    t11 = gemm(t10, weight_5)
    t12 = ReLU(t11)
    t13 = cross_entropy(t12, label)
  else:
    t1 = conv2d_nchw(img, weight_1, None, 1, 1, 1, 1)
    t3 = avgpool(t1)

    t4 = conv2d_nchw(t3, weight_2, None, 1, 0, 1, 1)
    t6 = avgpool(t4)

    t7 = flatten(t6)
    t8 = gemm(t7, weight_3)
    t9 = gemm(t8, weight_4)

    t11 = gemm(t9, weight_5)
    t12 = t11
    t13 = cross_entropy(t12, label)


  d1, d2, d3, d4, d5 = tvm.te.mygradient(t13, [weight_1, weight_2, weight_3, weight_4, weight_5])

  print("Build model...")

  s = tvm.te.create_schedule([t13.op, d1.op, d2.op, d3.op, d4.op, d5.op])

  print(tvm.lower(s, [img, label, weight_1, weight_2, weight_3, weight_4, weight_5, t12, t13, d1, d2, d3, d4, d5], simple_mode=True))

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

  ctx = tvm.context("llvm")

  img_tvm = tvm.nd.array(img_np, ctx)
  label_tvm = tvm.nd.array(label_np, ctx)
  logit_tvm = tvm.nd.array(logit_np, ctx)
  ret_tvm = tvm.nd.array(ret_np, ctx)
  free_vars_tvm = [tvm.nd.array(x, ctx) for x in params]
  gradients_tvm = [tvm.nd.array(x, ctx) for x in inits]

  func(img_tvm, label_tvm, *free_vars_tvm, logit_tvm, ret_tvm, *gradients_tvm)

  print("Start training...")
  # this will be updated during training
  model_weights = []
  for var in free_vars:
    shape = to_tuple(var.shape)
    var_np = np.random.uniform(-1, 1, shape).astype(dtype)
    model_weights.append(var_np)

  epoch = 3
  lr = 1e-6

  train_set = torchvision.datasets.MNIST(".", train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
  test_set = torchvision.datasets.MNIST(".", train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True)

  for ep in range(epoch):
    train_num_covered = 0
    running_acc = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader):
      img_tvm = tvm.nd.array(data[0].numpy().astype(dtype), ctx)
      label_torch = torch.tensor(np.zeros([batch, 10]).astype(dtype))
      label_torch.scatter_(1, data[1].unsqueeze(0).T, 1.0)
      #print("label_torch", label_torch)
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

      # accuracy & loss record
      train_num_covered += batch
      _, predict = torch.max(torch.from_numpy(logit_tvm.asnumpy()), 1)
      num_correct = (predict == data[1]).sum()
      running_acc += num_correct.item()
      running_loss += ret_tvm.asnumpy().item(0)

      if (i) % 1000 == 0:
        print("epoch=", ep+1, "iteration=", i+1, "loss=", running_loss/train_num_covered, "acc=", running_acc/train_num_covered)
        # print("logit=", logit_tvm.asnumpy()) #shape:[batch, 10]
        # print("model_weights[0]", model_weights[0])
        # #print("model_weights[4]", model_weights[4])
        # print("gradient_iter[0]", gradients_iter[0])
        # #print("gradient_iter[4]", gradients_iter[4])

      # print("Updating...")
      for k, gradient in enumerate(gradients_iter):
        assert(model_weights[k].shape == gradient.asnumpy().shape)
        model_weights[k] -= lr * gradient.asnumpy()

    assert(train_num_covered == len(train_set))
    running_acc /= len(train_set)
    print("epoch=", ep+1, "accuracy=", running_acc)
    
      


if __name__ == "__main__":
  main()
