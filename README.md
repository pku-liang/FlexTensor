# AutoScheduler for Tensor Computations

## Introductions

This auto-scheduler is used to automatically optimize tensor computations for CPU/GPU and is aiming at more targets in the future.
It relies on tensor compiler [TVM](https://github.com/dmlc/tvm) to generate codes for these targets.
Currently it uses profile-based method in searching for best schedules and plans to support more advanced methods in the future.


## Installation

Requires: `Python 3.5+`, `Numpy`

1. Install TVM, follow the [instructions](https://docs.tvm.ai/install/from_source.html).
2. Clone this repo:
   ```sh
   git clone https://github.com/KnowingNothing/AutoScheduler.git
   ```
3. Set the environments:
   `export AUTO_HOME=path/to/AutoScheduler`
   `export PYTHONPATH=$AUTO_HOME:$PYTHONPATH`

To run the baselines, `PyTorch` is required.



## Usage

### 1. Write compute in Python

For example, write a `gemm` kernel.
This uses TVM DSL in Python.
```py
import tvm

def gemm(A, B):
    """Matrix multiplies matrix

    Args:
    -----------------------------
    A: tvm.tensor.Tensor
        shape [height, width]
    B: tvm.tensor.Tensor
        shape [width, length]
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [height, length]
    -----------------------------
    """
    k = tvm.reduce_axis((0, B.shape[0]))
    return tvm.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k))
```
Existing computes for some common tensor computations are in `nn/ops.py`.

### 2. Register optimization tasks
Take the `gemm` kernel above as example.
First wrap the kernel.
```py
def wrap_gemm(N, K, M):
    A = tvm.placeholder((N, K))
    B = tvm.placeholder((K, M))
    Output = gemm(A, B)
    return [Output.op], [A, B, Output]
```
Then register a task.
```py
from auto_schedule.task import register_task, Task

'''
To create a task, the parameters are:
1. type of operator: str
2. name of this operator: str
3. the wrapper for tensor computation
4. arguments to the wrapper, i.e. input shapes
5. target device: str ("llvm" or "cuda" currently)
6. device number: int
'''
task = Task(
    "gemm", 
    "gemm", 
    wrap_gemm, 
    (1024, 1024, 1024), 
    "llvm", 
    0)
# register the task
register_task(task)
```

Existing tasks are registered in `task.py`.

### 3. Push the button

```py
from auto_schedule.scheduler import schedule

s, bufs, configs = schedule(
            task.key, # give the key of target task
            slevel=4,
            rlevel=3,
            op_trial=100, 
            timeout=10, 
            op_stop=30, 
            method="searching", 
            parallel=8,
            )
```
Wait patiently for the scheudling process to finish. This usually cost tens of miniutes on GPU and several hours on CPU. But it depends on the `op_trial`, `timeout`, `parallel` settings and input shapes.

### 4. Enjoy the results

The resulting `s` and `bufs` can be directly used to generate codes, the resulting `configs` can be saved to retrieve schedules.

```py
# directly use the results
func = tvm.build(s, bufs, task.target)
# use the configs
from auto_schedule.scheduler import schedule_with_config

s, bufs = schedule_with_config(task_key, configs)
func = tvm.build(s, bufs, task.target)
```