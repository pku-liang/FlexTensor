## Tested Ops

| op name | case No. | grad to | configs | gradient | build | correctness |
| --- | --- | --- | --- | --- | --- | --- |
| GEMM | 1 | A | | yes | yes | rtol=1e-5 |
| Conv2d | 1 | A | st=1, pad=0, group=1, dilation=1 | yes | yes | rtol=1e-3 |
| Conv2d | 2 | A | st=2, pad=0, group=1, dilation=1 | yes | yes | |
| Conv2d | 3 | A | st=2, pad=0, group=2, dilation=1 | yes | yes | |
| Flatten | 1 | A | | yes | yes | rtol < 1e-30 |
| Downcast | 1 | A | | yes | yes | rtol < 1e-30 |
| Broadcast | 1 | A | | yes | yes | rtol=1e-6 |
| Padding | 1 | A | | yes | yes | |
| AvgPool | 1 | A | | yes | yes | |
