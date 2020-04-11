import tvm
import numpy as np
from tvm import relay

import tvm.relay.testing
import tvm.contrib.graph_runtime as runtime

batch_size = 1

num_actions = 18

image_shape = (4, 84, 84)

target = "llvm"

dtype = "float32"

input_shape = (batch_size, *image_shape)

input_type = relay.TensorType(input_shape, dtype)

print("Get DQN network...")
# net = relay.testing.dqn.get_net(
#   batch_size, num_actions=num_actions, image_shape=image_shape, dtype=dtype)

fmod, fparams = relay.testing.dqn.get_workload(batch_size)

# print("Get gradient...")
# fmod = relay.transform.ToANormalForm()(fmod)
# bnet = relay.transform.gradient(fmod["main"], mode="higher_order")
# fmod = relay.transform.ToANormalForm()(fmod)

# print("Make workload")
# mod, params = relay.testing.create_workload(bnet)

print("Build graph...")
with relay.build_config(opt_level=1):
  graph, lib, params = relay.build_module.build(
      fmod, target=target, params=fparams)

ctx = tvm.context(str(target), 0)

print("Create runtime...")
module = runtime.create(graph, lib, ctx)

print("Set inputs...")
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input('data', data_tvm)
module.set_input(**params)

# module.run()

print(module.get_num_outputs())