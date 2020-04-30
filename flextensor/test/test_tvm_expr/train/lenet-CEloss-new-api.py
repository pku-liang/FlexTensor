from collections import OrderedDict

import tvm
import topi
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from flextensor.utils import to_tuple


class Module:

    def __init__(self, name):
        self.name = name

    def __call__(self, *inputs):
        raise NotImplementedError

    @property
    def weights(self):
        raise NotImplementedError


class Conv2d(Module):

    def __init__(self, name, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__(name)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        weight_shape = [out_channels, in_channels, kernel_size, kernel_size]
        self.weight = tvm.te.placeholder(weight_shape, dtype='float64', name=f'{name}_weight')
        if bias:
            self.bias = tvm.te.placeholder([out_channels, ], dtype='float64', name=f'{name}_bias')
        else:
            self.bias = None

    def __call__(self, inputs):
        outputs = topi.nn.conv2d(inputs, self.weight, self.stride, self.padding, self.dilation)
        if self.bias:  # TODO: check bias shape
            reshaped_bias = topi.reshape(self.bias, (self.in_channels, self.out_channels, 1, 1))
            outputs += reshaped_bias
        return outputs

    @property
    def weights(self):
        if self.bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]


def avg_pool2d(inputs, kernel_size=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]

    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = [stride, stride]

    if isinstance(padding, int):
        padding = [padding, ] * 4
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        padding = padding + padding

    return topi.nn.pool(
        inputs, kernel_size, stride, padding, 'avg',
        ceil_mode=ceil_mode, count_include_pad=count_include_pad,
    )


class Linear(Module):

    def __init__(self, name, in_features, out_features, bias=True):
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = tvm.te.placeholder([in_features, out_features], dtype='float64', name=f'{name}_weight')
        if bias:
            self.bias = tvm.te.placeholder([out_features, ], dtype='float64', name=f'{name}_bias')
        else:
            self.bias = None

    def __call__(self, inputs):
        return topi.nn.dense(inputs, self.weight, bias=self.bias)

    @property
    def weights(self):
        if self.bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]


# FIXME: Does not work
def flatten_topi(inputs):
    N, C, H, W = inputs.shape
    return topi.reshape(inputs, [N, C * H * W])


def flatten(inputs):
    """
    inputs: [batch, channel, height, width]
    return: [batch, channel * height * width]
    """
    assert(inputs.shape[1].value*inputs.shape[2].value*inputs.shape[3].value == 400)
    return tvm.te.compute([inputs.shape[0], inputs.shape[1]*inputs.shape[2]*inputs.shape[3]],
                          lambda i, j: inputs[i, j//(inputs.shape[2]*inputs.shape[3]),
                                              (j % (inputs.shape[2]*inputs.shape[3])) // inputs.shape[3],
                                              j % inputs.shape[3]]
                          , name="flatten", requires_grad=True)


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
    return tvm.te.compute([1], lambda i: tvm.te.sum(
        targets[i+rrn, rrc] * ((tvm.tir.log(sum_val[i+rrn])+max_val[i+rrn]) - inputs[i+rrn, rrc]*targets[i+rrn, rrc]) / N,
        axis=[rrn, rrc]), name="cross_entropy", requires_grad=True)


class Model:

    def __init__(self): pass

    def __call__(self, inputs):
        raise NotImplementedError

    @property
    def weights(self):
        raise NotImplementedError


disable_relu = True
if disable_relu:
    topi.nn.relu = lambda x: x


class LeNet(Model):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d('conv1', 1, 6, 3, stride=1, padding=1, bias=False)
        self.conv2 = Conv2d('conv2', 6, 16, 5, stride=1, padding=0, bias=False)
        self.fc1 = Linear('fc1', 400, 120, bias=False)
        self.fc2 = Linear('fc2', 120, 84, bias=False)
        self.fc3 = Linear('fc3', 84, 10, bias=False)

    def __call__(self, inputs):
        outputs = topi.nn.relu(self.conv1(inputs))
        outputs = avg_pool2d(outputs)
        outputs = topi.nn.relu(self.conv2(outputs))
        outputs = avg_pool2d(outputs)
        outputs = flatten(outputs)
        # outputs = topi.nn.relu(self.fc1(outputs))
        outputs = self.fc1(outputs)
        outputs = topi.nn.relu(self.fc2(outputs))
        outputs = topi.nn.relu(self.fc3(outputs))
        return outputs

    @property
    def weights(self):
        modules = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]
        return sum([m.weights for m in modules], [])


class Learner:

    def __init__(self, model, train_loader, num_classes, criterion, lr, print_freq=1000, target='llvm', dtype='float64'):
        self.model = model
        self.train_loader = train_loader
        self.num_classes = num_classes
        self.criterion = criterion
        self.lr = lr

        self.print_freq = print_freq
        self.target = target
        self.dtype = dtype
        self.ctx = tvm.context(target)

        self._build_func()
        self._allocate_buffers_for_endpoints()
        self._initialize_weights()

    def _build_func(self):
        images_pth, labels_pth = next(iter(self.train_loader))
        self.images = tvm.te.placeholder(list(images_pth.shape), dtype=self.dtype, name='images')
        self.labels = tvm.te.placeholder([labels_pth.shape[0], self.num_classes], dtype=self.dtype, name='labels')
        self.logit = self.model(self.images)
        self.loss = cross_entropy(self.logit, self.labels)
        self.gradients = tvm.te.mygradient(self.loss, model.weights)
        self.sched = tvm.te.create_schedule([self.loss.op] + [grad.op for grad in self.gradients])
        args = [self.images, self.labels, *self.model.weights, self.logit, self.loss, *self.gradients]
        # print(tvm.lower(self.sched, args, simple_mode=True))
        self.func = tvm.build(self.sched, args, target=self.target)

    def _allocate_buffers_for_endpoints(self):
        logit_np = np.zeros(to_tuple(self.logit.shape)).astype(self.dtype)
        loss_np = np.zeros(to_tuple(self.loss.shape)).astype(self.dtype)
        self.logit_tvm = tvm.nd.array(logit_np, self.ctx)
        self.loss_tvm = tvm.nd.array(loss_np, self.ctx)

    def _initialize_weights(self):
        self.weights_np = [np.random.uniform(-1, 1, to_tuple(var.shape)).astype(self.dtype) for var in self.model.weights]
        self.weights_tvm = [tvm.nd.array(var, self.ctx) for var in self.weights_np]

    def _preprocess_batch(self, images, targets):
        images_np = images.numpy().astype(dtype)
        images_tvm = tvm.nd.array(images_np, self.ctx)
        labels_np = np.zeros([batch_size, 10]).astype(dtype)
        labels_pth = torch.tensor(labels_np)
        labels_pth.scatter_(1, targets.unsqueeze(0).T, 1.0)
        labels_tvm = tvm.nd.array(labels_pth.numpy(), self.ctx)
        return images_tvm, labels_tvm

    def _reset_gradients(self):
        grads_np = [np.zeros(to_tuple(var.shape)).astype(self.dtype) for var in self.gradients]
        self.grads_tvm = [tvm.nd.array(var, self.ctx) for var in grads_np]

    def _execute_func(self, images_tvm, targets_tvm):
        self.func(images_tvm, targets_tvm, *self.weights_tvm, self.logit_tvm, self.loss_tvm, *self.grads_tvm)
        return self.logit_tvm.asnumpy(), self.loss_tvm.asnumpy().item(0)

    def _update_weights(self):
        for k, grad in enumerate(self.grads_tvm):
            assert(self.weights_np[k].shape == grad.asnumpy().shape)
            self.weights_np[k] -= self.lr * grad.asnumpy()
        self.weights_tvm = [tvm.nd.array(var, self.ctx) for var in self.weights_np]

    def _train_one_step(self, images, targets):
        batch_tvm = self._preprocess_batch(images, targets)
        self._reset_gradients()
        logit_np, loss_val = self._execute_func(*batch_tvm)
        preds = torch.from_numpy(np.argmax(logit_np, axis=1))
        self.running_acc += (preds == targets).sum().item()
        self.running_loss += loss_val
        self._update_weights()

    def train_one_epoch(self, epoch_idx):
        num_covered = 0
        self.running_acc = 0.0
        self.running_loss = 0.0
        for i, (images, targets) in enumerate(self.train_loader):
            num_covered += images.size()[0]
            self._train_one_step(images, targets)
            if i % self.print_freq == 0:
                loss_avg = self.running_loss / num_covered
                acc_avg = self.running_acc / num_covered
                print(f"epoch = {epoch_idx+1}, iteration = {i+1}: loss = {loss_avg}, acc = {acc_avg}")
        assert num_covered == len(self.train_loader.dataset)
        acc_avg = self.running_acc / num_covered
        print(f"epoch = {epoch_idx+1}: accuracy = {acc_avg}")

    def get_gradient(self, weight_key):
        for weight, grad in zip(self.model.weights, self.grads_tvm):
            if weight.name == weight_key:
                return grad.asnumpy()

    @property
    def state_dict(self):
        return OrderedDict({
            self.model.weights[k].name: self.weights_np[k]
            for k, grad in enumerate(self.grads_tvm)
        })


def load_mnist_dataset(batch_size, test_batch_size=1):
    train_set = torchvision.datasets.MNIST(".", train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    test_set = torchvision.datasets.MNIST(".", train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == '__main__':
    batch_size = 4
    lr = 1e-6
    num_epochs = 3
    num_classes = 10
    target = 'llvm'
    dtype = 'float64'

    model = LeNet()
    train_loader = load_mnist_dataset(batch_size)[0]
    criterion = cross_entropy
    learner = Learner(model, train_loader, num_classes, criterion, lr, target=target, dtype=dtype)

    for epoch_idx in range(num_epochs):
        learner.train_one_epoch(epoch_idx)

    print(learner.get_gradient('conv1_weight').shape)
