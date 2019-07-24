## 添加算子步骤

1. 将算子的实现部分添加在`.ops.py`，如
```python
def gemm(A, B, transposeA=False, transposeB=False):
    """Matrix multiplies matrix

    Args:
    -----------------------------
    A: tvm.tensor.Tensor
        shape [height, width]
    B: tvm.tensor.Tensor
        shape [width, length]
    transposeA: (optional:False) bool
    transposeB: (optional:False) bool
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [height, length]
    -----------------------------
    """
    if transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[0].value == B.shape[1].value)
        return tvm.compute((A.shape[1], B.shape[0]), lambda i, j: tvm.sum(A[k, i] * B[j, k], axis=k))
    elif transposeA and not transposeB:
        k = tvm.reduce_axis((0, B.shape[0]))
        assert_print(A.shape[0].value == B.shape[0].value)
        return tvm.compute((A.shape[1], B.shape[1]), lambda i, j: tvm.sum(A[k, i] * B[k, j], axis=k))
    elif not transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[1].value == B.shape[1].value)
        return tvm.compute((A.shape[0], B.shape[0]), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k))
    else:
        k = tvm.reduce_axis((0, B.shape[0]))
        assert_print(A.shape[1].value == B.shape[0].value)
        return tvm.compute((A.shape[0], B.shape[1]), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k))
```
2. 在`../configs/`中添加配置文件，如`../configs/gemm_config.py`

```python
gemm_shapes = [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048)
]
```
3. 在`../task.py`中注册算子的task，如

```python
from auto_schedule.nn.ops import gemm as op_gemm

def gemm(N, K, M):
    A = tvm.placeholder((N, K))
    B = tvm.placeholder((K, M))
    Output = op_gemm(A, B)
    return [Output.op], [A, B, Output]

from auto_schedule.configs.gemm_config import gemm_shapes

for shape in gemm_shapes:
    N, K, M = shape
    for j in range(4):
        register_task(Task("gemm", "gemm", gemm, (N, K, M), "llvm", j))
        register_task(Task("gemm", "gemm", gemm, (N, K, M), "cuda", j))
```
4. （可选）在`../test/test_ops.py`中添加算子的正确性测试
5. 在`../optimize/`中添加优化测试