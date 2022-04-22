import tvm
import topi
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from flextensor.utils import to_tuple, assert_print

print_per_iteration = 100
debug_mode = False
debug_print_cnt = 2
CLIP_VALUE = 1e20
batch_size = 100
learning_rate = 1
num_epoches = 3
state_size = 128
dtype = "float64"
target_platform = "llvm"


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def numpy_init(weight_list,*args):
  '''
  the first argument is randomly initialized. All others are zero initialized.
  '''
  weight_np = [np.random.uniform(-1, 1, to_tuple(var.shape)).astype(dtype) for var in weight_list]
  init = [weight_np]
  if len(args) > 0:
    for item in args:
      init.append([np.zeros(to_tuple(var.shape), dtype=dtype) for var in item])
  return init

def elu(inputs):
  '''
  https://pytorch.org/docs/stable/nn.html#torch.nn.ELU
  ELU(x)=max(0,x)+min(0,(exp(x)−1))
  '''
  func = lambda i, j: tvm.te.if_then_else(inputs[i, j] > 0, inputs[i, j], tvm.tir.exp(inputs[i, j]) - 1)
  return tvm.te.compute(inputs.shape, func, name="elu")

def cross_entropy(inputs, targets):
  '''
  https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
  loss(x,class) = −x[class]+log(\Sigma:j exp(x[j]))
  -x[class]: since targets is one-hot, we use "inner-dot" to compute x_class
  log(\Sigma:j exp(x[j])) may overflow when computing exp(x[j])
  >>>>Trick: maxval + log(\Sigma: j exp(x[j] - maxval))
  Finally, compute the average over batches
  '''
  assert_print(inputs.shape[0].value == targets.shape[0].value)
  assert_print(inputs.shape[1].value == targets.shape[1].value)
  N, C = inputs.shape
  c = tvm.te.reduce_axis([0, C], "c")
  k1 = tvm.te.reduce_axis([0, C], name="k1")
  # First compute the maximum for each batch
  max_val = tvm.te.compute([N], lambda n: tvm.te.max(inputs[n, k1], axis=[k1]), name="max_val")
  # Use the log_softmax trick to avoid overflow
  sum_val = tvm.te.compute([N], lambda i: tvm.te.sum(tvm.tir.exp(inputs[i, c]-max_val[i]), axis=[c]), "sum_val")
  rrn = tvm.te.reduce_axis([0, N], "rrn")
  rrc = tvm.te.reduce_axis([0, C], "rrc")
  x_class = tvm.te.compute([N], lambda i: tvm.te.sum(inputs[i, rrc]*targets[i, rrc], axis=[rrc]), name="x_class")
  return tvm.te.compute([1],
    lambda i: tvm.te.sum((tvm.tir.log(sum_val[i+rrn])+max_val[i+rrn] - x_class[i+rrn]) / N, axis=[rrn]), name="cross_entropy")

def internel_lltm(input, weight_for_gate, bias_for_gate, old_h, old_c):
  '''
  input: [batch_size, 28*28]
  old_h  & old_c: [batch_size, state_size]
  >>>>> cat -> X: [batch_size, state_size+28*28]
  weight_for_gate: [3*state_size, state_size+28*28]
  bias_for_gate:[3*state_size]
  '''
  X = topi.concatenate([old_h, input], axis=1)
  gate_weights = topi.nn.dense(X, weight_for_gate, bias_for_gate)
  gates = topi.split(gate_weights, 3, axis=1)
  input_gate = topi.sigmoid(gates[0])
  output_gate = topi.sigmoid(gates[1])
  candidate_cell = elu(gates[2])
  new_c = topi.add(old_c, topi.multiply(candidate_cell, input_gate))
  new_h = topi.multiply(topi.tanh(new_c), output_gate)
  return [new_h, new_c]

def lltm(input, targets, weight_for_classify, bias_for_classify, weight_for_gate, bias_for_gate, old_h, old_c):
  '''
  input: [batch_size, 28*28]
  new/old_h & new/old_c: [batch_size, state_size]
  weight_for_classify: [10, state_size]
  bias_for_classify: [10]
  result: [batch_size, 10]
  targets: [batch_size, 10] one-hot
  '''
  new_h, new_c = internel_lltm(input, weight_for_gate, bias_for_gate, old_h, old_c)
  assert_print(new_h.shape[1].value == weight_for_classify.shape[1].value)
  result = topi.nn.dense(new_h, weight_for_classify, bias_for_classify)
  loss = cross_entropy(result, targets)
  return loss, result, new_h, new_c

def main():
  global debug_print_cnt
  img = tvm.te.placeholder([batch_size, 28*28], dtype=dtype, name="img")
  label = tvm.te.placeholder([batch_size, 10], dtype=dtype, name="label")

  weight_for_classify = tvm.te.placeholder([10, state_size], dtype=dtype, name="weight_for_classify")
  bias_for_classify = tvm.te.placeholder([10], dtype=dtype, name="bias_for_classify")
  weight_for_gate = tvm.te.placeholder([3*state_size, state_size+28*28], dtype=dtype, name="weight_for_gate")
  bias_for_gate = tvm.te.placeholder([3*state_size], dtype=dtype, name="bias_for_gate")

  old_h = tvm.te.placeholder([batch_size, state_size], dtype=dtype, name="old_h") #, requires_grad=False)
  old_c = tvm.te.placeholder([batch_size, state_size], dtype=dtype, name="old_c") #, requires_grad=False)

  # Helper list
  weight_to_update = [weight_for_classify, bias_for_classify, weight_for_gate, bias_for_gate]
  old_hc = [old_h, old_c]
  
  #Function
  loss, result, new_h, new_c = lltm(img, label, *weight_to_update, *old_hc)

  #Helper list
  loss_and_result = [loss, result]
  new_hc = [new_h, new_c]
  grad_list = tvm.te.mygradient(loss, weight_to_update)

  s = tvm.te.create_schedule([var.op for var in loss_and_result] + [var.op for var in new_hc] + [grad.op for grad in grad_list])
  print(tvm.lower(s, [img, label, *weight_to_update, *old_hc, *loss_and_result, *new_hc, *grad_list], simple_mode=True))
  func = tvm.build(s, [img, label, *weight_to_update, *old_hc, *loss_and_result, *new_hc, *grad_list], target= target_platform)

  weight_np, old_hc_np, loss_result_np, new_hc_np, grad_np = numpy_init(weight_to_update, old_hc, loss_and_result, new_hc, grad_list)
  ctx = tvm.device(target_platform)

  for ep in range(num_epoches):
    train_num_covered = 0
    running_acc = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader):
      img_tvm = tvm.nd.array(data[0].squeeze(1).view(batch_size, 28*28).numpy().astype(dtype), ctx)
      label_torch = torch.tensor(np.zeros([batch_size, 10]).astype(dtype))
      label_torch.scatter_(1, data[1].unsqueeze(0).T, 1.0)
      #print("label_torch", label_torch)
      label_tvm = tvm.nd.array(label_torch.numpy(), ctx)
      weight_tvm = [tvm.nd.array(var) for var in weight_np]
      old_hc_tvm = [tvm.nd.array(var) for var in old_hc_np]
      loss_result_tvm = [tvm.nd.array(var) for var in loss_result_np]
      new_hc_tvm = [tvm.nd.array(var) for var in new_hc_np]
      grad_tvm = [tvm.nd.array(var) for var in grad_np]
      if debug_mode:
        print("before func, loss_result_tvm", loss_result_tvm)
        print("before func, img_tvm", img_tvm)
        print("before func, label_tvm", label_tvm)
        print("before func, weight_tvm", weight_tvm)
        print("before func, old_hc_tvm", old_hc_np)
        print("before func, new_hc_tvm", new_hc_tvm)
        print("before func, grad_tvm", grad_tvm)
      func(img_tvm, label_tvm, *weight_tvm, *old_hc_tvm, *loss_result_tvm, *new_hc_tvm, *grad_tvm)
      if debug_mode:
        print("after func, loss_result_tvm", loss_result_tvm)
        print("after func, img_tvm", img_tvm)
        print("after func, label_tvm", label_tvm)
        print("after func, weight_tvm", weight_tvm)
        print("after func, old_hc_tvm", old_hc_np)
        print("after func, new_hc_tvm", new_hc_tvm)
        print("after func, grad_tvm", grad_tvm)
        debug_print_cnt = debug_print_cnt - 1
        if debug_print_cnt == 0:
          exit(0)

      train_num_covered += batch_size
      # loss_result_tvm is a list: 
      # >>>>>> loss:[1], result:[batch_size, 10]
      _, predict = torch.max(torch.from_numpy(loss_result_tvm[1].asnumpy()), 1)
      num_correct = (predict == data[1]).sum()
      running_acc += num_correct.item()
      running_loss += loss_result_tvm[0].asnumpy().item(0)

      if i % print_per_iteration == 0:
        print("epoch=", ep+1, "iteration=", i+1, "loss=", running_loss/train_num_covered, "acc=", running_acc/train_num_covered)
      
      for k, gradient in enumerate(grad_tvm):
        assert(weight_np[k].shape == gradient.asnumpy().shape)
        gradient_clipped = np.clip(np.nan_to_num(gradient.asnumpy(), nan=CLIP_VALUE), -CLIP_VALUE, CLIP_VALUE)
        weight_np[k] -= learning_rate * gradient_clipped
      
      #we update hidden_states and cell_states for next iteration
      for k, item in enumerate(new_hc_tvm):
        assert(old_hc_np[k].shape == item.asnumpy().shape)
        item_clipped = np.clip(np.nan_to_num(item.asnumpy(), nan=CLIP_VALUE), -CLIP_VALUE, CLIP_VALUE)
        old_hc_np[k] = item_clipped
      
      #zero the new hidden_states and new cell_states
      for k, item in enumerate(new_hc_np):
        new_hc_np[k] = np.zeros(new_hc_np[k].shape, dtype=dtype)
          
    assert(train_num_covered == len(train_dataset))
    running_acc /= len(train_dataset)
    print("epoch=", ep+1, "accuracy=", running_acc)


if __name__ == "__main__":
  main()
