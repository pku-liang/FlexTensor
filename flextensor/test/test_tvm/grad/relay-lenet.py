import tvm
import numpy as np
from tvm import relay
import tvm.contrib.graph_runtime as runtime
import layers
import tvm.relay.testing

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
trans_img = transforms.ToTensor()

trainset = MNIST('./data', train=True, transform=trans_img, download=True)
testset = MNIST('./data', train=False, transform=trans_img, download=True)

batch_size = 1
image_shape = (1, 28, 28)
target = "cuda"
dtype = "float32"
epoches = 1

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

def get_LeNet(batch_size=batch_size, img_shape=(1, 28, 28), dtype="float32"):
    data_shape = (batch_size,) + img_shape
    data = relay.var("data", shape=data_shape, dtype=dtype)
    conv1_bias = relay.var("conv1_bias")
    conv1 = layers.conv2d(data, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), channels=6, name="conv1")
    conv1 = relay.nn.bias_add(conv1, conv1_bias)
    maxpool1 = relay.nn.max_pool2d(conv1, (2, 2), (2, 2))
    conv2_bias = relay.var("conv2_bias")
    conv2 = layers.conv2d(maxpool1, kernel_size=(5, 5), strides=(1, 1), padding=(0, 0), channels=16, name="conv2")
    conv2 = relay.nn.bias_add(conv2, conv2_bias)
    maxpool2 = relay.nn.max_pool2d(conv2, (2, 2), (2, 2))
    bf1 = relay.nn.batch_flatten(maxpool2)
    dense1 = layers.dense_without_bias(bf1, units=120, name="dense1")
    dense2 = layers.dense_without_bias(dense1, units=84, name="dense2")
    dense3 = layers.dense_without_bias(dense2, units=10, name="dense3")
    softmax = relay.nn.softmax(dense3)
    #label is from input
    label = relay.var("data2", shape=(batch_size, 10), dtype=dtype)
    loss = relay.nn.cross_entropy(softmax, label)
    args = relay.analysis.free_vars(loss)
    return relay.Function(args, loss)
def get_workload():
    net = get_LeNet()
    return tvm.relay.testing.create_workload_with_label(net)
print("Get LeNet network...")
net = get_LeNet()
fmod, fparams = get_workload()

print("Get gradient...")
bnet = relay.transform.gradient(fmod["main"], mode='first_order')

print("Make workload")
mod, params = relay.testing.create_workload_with_label(bnet)
#mod, params = relay.testing.create_workload_with_label(net) #will be good for forward network

print("Build graph...")
with relay.build_config(opt_level=1):
  graph, lib, params = relay.build_module.build(
      mod, target=target, params=params)

ctx = tvm.context(str(target), 0)

print("Create runtime...")
module = runtime.create(graph, lib, ctx)

print("Set inputs...")
data_tvm = tvm.nd.array((np.random.uniform(size=(batch_size, 1, 28, 28))).astype(dtype))
label_tvm = tvm.nd.array(np.array([np.eye(10)[2]]).astype(dtype))
module.set_input('data', data_tvm)
module.set_input('data2', label_tvm)
module.set_input(**params)

module.run()

print(f'#outputs: {module.get_num_outputs()}') # 9
relay_output = module.get_output(0).asnumpy()
print(relay_output, relay_output.shape)
#[[ 0.01223469 -0.6366852  -0.07976073 -0.33273488 -0.41739434  0.18242726  0.5345733  -0.23996258 -0.4879382   0.14173466]]

for output_idx in range(2, 9):
    output = module.get_output(output_idx)
    print(f'Shape: {output.shape}')
    # Shape: (1, 1, 28, 28) -> input
    # Shape: (6, 1, 3, 3) -> p0  output_idx=2
    # Shape: (6,) -> p1
    # Shape: (16, 6, 5, 5) -> p2
    # Shape: (16,)   -> p3
    # Shape: (120, 400) -> p4
    # Shape: (84, 120) -> p5
    # Shape: (10, 84) -> p6
    # Shape: (1, 10) -> label output_idx = 9
    params['p' + str(output_idx - 2)] = tvm.nd.array((params['p' + str(output_idx - 2)].asnumpy() - 0.00001 * output.asnumpy()).astype(dtype))
for i in range(epoches):
    for (img, label) in trainloader:
        img_tvm = tvm.nd.array(img.numpy().astype(dtype))
        label_tvm = tvm.nd.array(np.array([np.eye(10)[int(label)]]).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input('data2', label_tvm)
        module.set_input(**params)
        module.run()
        print("loss:", module.get_output(0).asnumpy())
        for output_idx in range(2, 9):
            output = module.get_output(output_idx)
            #print(f'Shape: {output.shape}')
            params['p' + str(output_idx - 2)] = tvm.nd.array((params['p' + str(output_idx - 2)].asnumpy() - 0.01 * output.asnumpy()).astype(dtype))
