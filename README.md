# FlexTensor

### FlexTensor is moving to https://github.com/pku-liang

## Introductions

Tensor computation plays a paramount role in a broad range
of domains, including machine learning, data analytics, and
scientific computing. The wide adoption of tensor computation
and its huge computation cost has led to high demand
for flexible, portable, and high-performance library implementation
on heterogeneous hardware accelerators such
as GPUs and FPGAs. However, the current tensor library
implementation mainly requires programmers to manually
design low-level implementation and optimize from the algorithm,
architecture, and compilation perspectives. Such
a manual development process often takes months or even
years, which falls far behind the rapid evolution of the application
algorithms.

We introduce FlexTensor, which is a schedule
exploration and optimization framework for tensor computation
on heterogeneous systems. FlexTensor can optimize
tensor computation programs without human interference,
allowing programmers to only work on high-level programming
abstraction without considering the hardware platform
details. FlexTensor systematically explores the optimization
design spaces that are composed of many different schedules
for different hardware. Then, FlexTensor combines different
exploration techniques, including heuristic method and 
machine learning method to find the optimized schedule
configuration. Finally, based on the results of exploration,
customized schedules are automatically generated for different
hardware. In the experiments, we test 12 different kinds
of tensor computations with totally hundreds of test cases
and FlexTensor achieves average 1.83x performance speedup
on NVIDIA V100 GPU compared to cuDNN; 1.72x performance
speedup on Intel Xeon CPU compared to MKL-DNN
for 2D convolution; 1.5x performance speedup on Xilinx
VU9P FPGA compared to OpenCL baselines; 2.21x speedup
on NVIDIA V100 GPU compared to the state-of-the-art.


## Installation

Requires: `Python 3.5+`, `Numpy`, `tvm: https://github.com/KnowingNothing/tvm/tree/mirror`

1. Install TVM, follow the [instructions](https://docs.tvm.ai/install/from_source.html).
2. Clone this repo:
   ```sh
   git clone https://github.com/KnowingNothing/FlexTensor.git
   ```
3. Set the environments:
   `export AUTO_HOME=path/to/FlexTensor`
   `export PYTHONPATH=$AUTO_HOME:$PYTHONPATH`

To run the baselines, `PyTorch` is required.



## Usage

### 1. Write compute in Python

For example, write a `gemm` kernel.
This uses TVM DSL in Python.
```python
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

```python
def wrap_gemm(N, K, M):
    A = tvm.placeholder((N, K))
    B = tvm.placeholder((K, M))
    Output = gemm(A, B)
    return [Output.op], [A, B, Output]
```
Then register a task.
```python
from flextensor.task import register_task, Task

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

```python
from flextensor.scheduler import schedule

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

```python
# directly use the results
func = tvm.build(s, bufs, task.target)
# use the configs
from flextensor.scheduler import schedule_with_config

s, bufs = schedule_with_config(task_key, configs)
func = tvm.build(s, bufs, task.target)
```

## Citing FlexTensor
If you find FlexTensor useful for your project, please cite the following paper：

@inproceedings{FlexTensor,
  author    = {Size Zheng and
               Yun Liang and
               Shuo Wang and
               Renze Chen and
               Kaiwen Sheng},
  editor    = {James R. Larus and
               Luis Ceze and
               Karin Strauss},
  title     = {FlexTensor: An Automatic Schedule Exploration and Optimization Framework
               for Tensor Computation on Heterogeneous System},
  booktitle = {{ASPLOS} '20: Architectural Support for Programming Languages and
               Operating Systems, Lausanne, Switzerland, March 16-20, 2020 {[ASPLOS}
               2020 was canceled because of {COVID-19]}},
  pages     = {859--873},
  publisher = {{ACM}},
  year      = {2020},
  url       = {https://doi.org/10.1145/3373376.3378508},
  doi       = {10.1145/3373376.3378508},
  timestamp = {Mon, 16 Mar 2020 11:14:36 +0100},
  biburl    = {https://dblp.org/rec/conf/asplos/Zheng0WCS20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
