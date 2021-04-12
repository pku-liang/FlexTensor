import json
from collections import OrderedDict
from typing import Union, Callable

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import tvm
from flextensor.utils import to_tuple

import topi


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


def relu(x):
    """Take relu of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return tvm.te.compute(x.shape, lambda *i: tvm.te.max(x(*i), tvm.tir.const(0, x.dtype)))


class Linear(Module):

    def __init__(self, name, in_features, out_features, bias=True):
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        # https://docs.tvm.ai/api/python/topi.html#topi.nn.dense
        # - weight (tvm.te.Tensor) – 2-D with shape [out_dim, in_dim]
        self.weight = tvm.te.placeholder([out_features, in_features], dtype='float64', name=f'{name}_weight')
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


class LeNet(Model):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d('conv1', 1, 6, 3, stride=1, padding=1, bias=False)
        self.conv2 = Conv2d('conv2', 6, 16, 5, stride=1, padding=0, bias=False)
        self.fc1 = Linear('fc1', 400, 120, bias=False)
        self.fc2 = Linear('fc2', 120, 84, bias=False)
        self.fc3 = Linear('fc3', 84, 10, bias=False)

    def __call__(self, inputs, debug_mode=False):
        debug_tensors = OrderedDict()

        def D(tensor, key):
            if debug_mode:
                debug_tensors[key] = tensor
            return tensor

        outputs = D(self.conv1(inputs), 'conv1')
        outputs = D(relu(outputs), 'relu1')
        outputs = D(avg_pool2d(outputs), 'pool1')
        outputs = D(self.conv2(outputs), 'conv2')
        outputs = D(relu(outputs), 'relu2')
        outputs = D(avg_pool2d(outputs), 'pool2')
        outputs = D(flatten(outputs), 'flatten')
        # outputs = topi.nn.relu(self.fc1(outputs))
        outputs = D(self.fc1(outputs), 'fc1')
        outputs = D(self.fc2(outputs), 'fc2')
        outputs = D(relu(outputs), 'relu3')
        outputs = D(self.fc3(outputs), 'fc3')
        outputs = relu(outputs)

        if debug_mode: return outputs, debug_tensors  # OrderedDict({'relu3': debug_tensors['relu3']})
        else: return outputs

    @property
    def weights(self):
        modules = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]
        return sum([m.weights for m in modules], [])


class MLP(Model):

    def __init__(self):
        super().__init__()
        self.fc = Linear('fc3', 28*28, 10, bias=False)

    def __call__(self, inputs, debug_mode=False):
        debug_tensors = OrderedDict()

        def D(tensor, key):
            if debug_mode:
                debug_tensors[key] = tensor
            return tensor

        outputs = D(flatten(inputs), 'flatten')
        outputs = self.fc(outputs)

        if debug_mode: return outputs, debug_tensors
        else: return outputs

    @property
    def weights(self):
        modules = [self.fc]
        return sum([m.weights for m in modules], [])


class Learner:

    # noinspection PyProtectedMember
    def __init__(self, model, train_loader, num_classes, criterion,
                 lr: Union[float, Callable[[int], float]],  # lr(epoch,) -> lr
                 debug_mode=False, print_freq=1000, target='llvm', dtype='float64'):
        self.model = model
        self.train_loader = train_loader
        self.num_classes = num_classes
        self.criterion = criterion
        self.lr = lr if isinstance(lr, float) else lr(0)
        self._lr_func = lr if not isinstance(lr, float) else lambda epoch: lr

        self.debug_mode = debug_mode
        self.print_freq = print_freq
        self.target = target
        self.dtype = dtype
        self.ctx = tvm.device(target)

        self._build_func()
        self._allocate_buffers_for_endpoints()
        self._initialize_weights()

    def _build_func(self):
        images_pth, labels_pth = next(iter(self.train_loader))
        self.images = tvm.te.placeholder(list(images_pth.shape), dtype=self.dtype, name='images')
        self.labels = tvm.te.placeholder([labels_pth.shape[0], self.num_classes], dtype=self.dtype, name='labels')
        if self.debug_mode:
            self.logit, self.debug_tensors = self.model(self.images, debug_mode=self.debug_mode)
        else:
            self.logit = self.model(self.images, debug_mode=self.debug_mode)
        self.loss = cross_entropy(self.logit, self.labels)
        self.gradients = tvm.te.mygradient(self.loss, model.weights)
        extra_args = list(self.debug_tensors.values()) if self.debug_mode else list()
        self.sched = tvm.te.create_schedule([self.loss.op] + [tensor.op for tensor in extra_args] + [grad.op for grad in self.gradients])
        args = [self.images, self.labels, *self.model.weights, self.logit, self.loss, *extra_args, *self.gradients]
        # print(tvm.lower(self.sched, args, simple_mode=True))
        self.func = tvm.build(self.sched, args, target=self.target)

    def _allocate_buffers_for_endpoints(self):
        def create_buffer(tensor):
            np_buffer = np.zeros(to_tuple(tensor.shape)).astype(self.dtype)
            tvm_buffer = tvm.nd.array(np_buffer, self.ctx)
            return tvm_buffer

        self.logit_tvm = create_buffer(self.logit)
        self.loss_tvm = create_buffer(self.loss)
        if self.debug_mode:
            self.debug_tensors_tvm = {
                key: create_buffer(tensor)
                for key, tensor in self.debug_tensors.items()
            }
        else:
            self.debug_tensors_tvm = {}

    def _initialize_weights(self):

        # TODO: support BatchNorm2d
        # NOTE: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py#L49-L60
        def init_weight(var):
            w_pth = torch.empty(*to_tuple(var.shape), dtype=torch.float64)
            if len(w_pth.shape) == 4:  # Conv2d
                # NOTE: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
                torch.nn.init.kaiming_normal_(w_pth, mode='fan_out', nonlinearity='relu')
            elif len(w_pth.shape) == 2:  # Linear
                torch.nn.init.normal_(w_pth, mean=0, std=0.01)
            elif len(w_pth.shape) == 1:  # bias
                torch.nn.init.constant_(w_pth, 0)
            else:
                raise NotImplementedError(f'Unrecognized weight shape: {var.shape}')
            return w_pth.numpy()

        self.weights_np = [init_weight(var) for var in self.model.weights]
        # self.weights_np = [np.random.uniform(-1, 1, to_tuple(var.shape)).astype(self.dtype) for var in self.model.weights]
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
        self.func(
            images_tvm, targets_tvm, *self.weights_tvm, self.logit_tvm, self.loss_tvm,
            *self.debug_tensors_tvm.values(), *self.grads_tvm
        )
        debug_tensors_np = {key: tvm_array.asnumpy() for key, tvm_array in self.debug_tensors_tvm.items()}
        if not self.debug_mode:
            return self.logit_tvm.asnumpy(), self.loss_tvm.asnumpy().item(0)
        else:
            return self.logit_tvm.asnumpy(), self.loss_tvm.asnumpy().item(0), debug_tensors_np

    def _update_weights(self):
        for k, grad in enumerate(self.grads_tvm):
            assert(self.weights_np[k].shape == grad.asnumpy().shape)
            self.weights_np[k] -= self.lr * grad.asnumpy()
        self.weights_tvm = [tvm.nd.array(var, self.ctx) for var in self.weights_np]

    def _train_one_step(self, images, targets, record=True):
        batch_tvm = self._preprocess_batch(images, targets)
        self._reset_gradients()
        logit_np, loss_val, *debug_tensors_np = self._execute_func(*batch_tvm)
        if self.debug_mode:
            self.debug_tensors_np = debug_tensors_np[0]
            self.debug_tensors_np.update({ 'logit': logit_np, 'loss': loss_val, })
        else:
            self.debug_tensors_np = dict()

        preds = torch.from_numpy(np.argmax(logit_np, axis=1))
        if record:
            self.running_acc += (preds == targets).sum().item()
            self.running_loss += loss_val * images.size()[0]
        self._update_weights()

    def train_one_epoch(self, epoch_idx):
        num_covered = 0
        self.running_acc = 0.0
        self.running_loss = 0.0
        self.lr = self._lr_func(epoch_idx)
        for i, (images, targets) in enumerate(self.train_loader):
            num_covered += images.size()[0]
            self._train_one_step(images, targets)
            if i % self.print_freq == 0:
                loss_avg = self.running_loss / num_covered
                acc_avg = self.running_acc / num_covered
                print(f"epoch = {epoch_idx+1}, iteration = {i+1}: lr = {self.lr}, loss = {loss_avg}, acc = {acc_avg}")
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
            weight.name: weight_np
            for weight, weight_np in zip(self.model.weights, self.weights_np)
        })

    @property
    def grads_dict(self):
        return OrderedDict({
            weight.name: grad.asnumpy()
            for weight, grad in zip(self.model.weights, self.grads_tvm)
        })

    @property
    def debug_dict(self):
        return self.debug_tensors_np


def load_mnist_dataset(batch_size, test_batch_size=1):
    train_set = torchvision.datasets.MNIST(".", train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    test_set = torchvision.datasets.MNIST(".", train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def pprint_dict(d):
    return json.dumps(d, indent=2)


if __name__ == '__main__':
    batch_size = 4
    # lr = 1e-6
    lr = lambda epoch: [1e-2, 1e-3, 1e-4][epoch]  # learning rate scheduler
    num_epochs = 3
    num_classes = 10
    target = 'llvm'
    dtype = 'float64'

    model = LeNet()
    train_loader = load_mnist_dataset(batch_size)[0]
    criterion = cross_entropy
    learner = Learner(model, train_loader, num_classes, criterion, lr, debug_mode=True, target=target, dtype=dtype)

    state_dict = {
        key: (nparray.min(), nparray.max())
        for key, nparray in learner.state_dict.items()
    }
    print('state_dict:', pprint_dict(state_dict))

    images, targets = next(iter(train_loader))
    # noinspection PyProtectedMember
    learner._train_one_step(images, targets, record=False)
    debug_dict = {
        key: (nparray.min(), nparray.max()) if not isinstance(nparray, float) else nparray
        for key, nparray in learner.debug_dict.items()
    }
    grads_dict = {
        key: (nparray.min(), nparray.max()) if not isinstance(nparray, float) else nparray
        for key, nparray in learner.grads_dict.items()
    }

    print('debug_dict:', pprint_dict(debug_dict))
    print('grads_dict:', pprint_dict(grads_dict))

    for epoch_idx in range(num_epochs):
        learner.train_one_epoch(epoch_idx)

    print(learner.get_gradient('conv1_weight').shape)
