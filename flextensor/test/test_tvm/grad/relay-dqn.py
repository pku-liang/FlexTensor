import tvm
import numpy as np
import torch
from tvm import relay

import tvm.relay.testing
import tvm.contrib.graph_runtime as runtime

from dqn_pytorch import DQN

batch_size = 1

num_actions = 18

image_shape = (4, 84, 84)

target = "llvm"

dtype = "float32"

input_shape = (batch_size, *image_shape)

input_type = relay.TensorType(input_shape, dtype)

print("Get DQN network...")
net = relay.testing.dqn.get_net(
  batch_size, num_actions=num_actions, image_shape=image_shape, dtype=dtype)

fmod, fparams = relay.testing.dqn.get_workload(batch_size)

print("Get gradient...")
bnet = relay.transform.gradient(fmod["main"], mode='first_order')  # default: higher_order

print("Make workload")
mod, params = relay.testing.create_workload(bnet)  # print(mod.get_global_vars())  # [GlobalVar(main)]

pytorch_model = DQN(num_actions=num_actions, image_shape=image_shape)
param_name_mapping = {
    'conv1.weight': 'conv1_weight',
    'conv1.bias': 'conv1_bias',
    'conv2.weight': 'conv2_weight',
    'conv2.bias': 'conv2_bias',
    'conv3.weight': 'conv3_weight',
    'conv3.bias': 'conv3_bias',
    'fc1.weight': 'dense1_weight',
    'fc1.bias': 'dense1_bias',
    'fc2.weight': 'dense2_weight',
    'fc2.bias': 'dense2_bias',
}
pytorch_model.load_state_dict({
    pth_key: torch.from_numpy(params[tvm_key].asnumpy())
    for pth_key, tvm_key in param_name_mapping.items()
}, strict=True)
pytorch_model.train()

print("Build graph...")
with relay.build_config(opt_level=3):
  graph, lib, params = relay.build_module.build(
      mod, target=target, params=params)

ctx = tvm.device(str(target), 0)

print("Create runtime...")
module = runtime.create(graph, lib, ctx)

print("Set inputs...")
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input('data', data_tvm)
module.set_input(**params)

module.run()

print(f'#outputs: {module.get_num_outputs()}')

relay_output = module.get_output(0).asnumpy()
pytorch_output = pytorch_model(torch.from_numpy(data_tvm.asnumpy()))
pytorch_output_np = pytorch_output.data.numpy()
pytorch_output.sum().backward()

for output_idx in range(1, 12):
    output = module.get_output(output_idx)
    print(f'Shape: {output.shape}', end=' ')
    print(f'Mean: {output.asnumpy().mean()}')

for name, param in pytorch_model.named_parameters():
    print(f'{name}: {param.grad.mean().item()}')

print(f'Allclose: {np.allclose(relay_output, pytorch_output_np)}')