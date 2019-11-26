from __future__ import absolute_import

import time
import json
import tvm
from auto_schedule.task import register_task, Task
from auto_schedule.measure import _evaluate
from auto_schedule.nn import conv2d_nchwc
from auto_schedule.configs.conv2d_config import yolo_shapes_b1
from auto_schedule.scheduler import schedule, schedule_with_config
from auto_schedule.utils import RpcInfo


def conv2d_nchwc_compute_avx2(N, C, H, W, K, k=3, use_bias=False, st=1, pad=0, dilation=1, group=1, vlen1=8, vlen2=8):
    inputs = tvm.placeholder([N, C // vlen1 // group, H, W, vlen1], dtype="float32")
    weight = tvm.placeholder([K // vlen2, C // vlen1 // group, k, k, vlen1, vlen2], dtype="float32")
    if use_bias:
        bias = tvm.placeholder([K // vlen2, vlen2], dtype="float32")
    else:
        bias = None 
    output = conv2d_nchwc(inputs, weight, bias, stride=st, padding=pad, dilation=dilation, groups=group)
    if use_bias:
        return output, [inputs, weight, bias, output]
    else:
        return [output.op], [inputs, weight, output]


if __name__ == "__main__":
    N, C, H, W, K, _, k, _, _, st, pad, dilation, group = yolo_shapes_b1[5]

    use_bias = False
    vlen = 8
    target = "llvm"
    dev_id = 0
    trials = 100
    timeout = 10
    parallel = 20
    method = "searching"
    force_inline = True 
    use_model = False
    logfile = open("tmp.log", "w")
    rpc_info = RpcInfo("0.0.0.0", 9090, target_host="llvm")

    args = (N, C, H, W, K, k, use_bias, st, pad, dilation, group)
    task = Task("conv2d_nchwc", "yolo_conv6", conv2d_nchwc_compute_avx2, args, target, dev_id=dev_id)
    register_task(task, override=False)

    beg = time.time()
    s, bufs, configs = schedule(
            task.key, 
            op_trial=trials, 
            timeout=timeout, 
            op_stop=30, 
            parallel=parallel, 
            method=method,
            use_model=use_model,
            trials=[trials//10, trials],
            force_inline=force_inline,
            rpc_info=rpc_info,
            slevel=2,
            rlevel=2
            )
    end = time.time()

    print("######################################")
    print("op schedules:")
    for config in configs.op_config_lst:
        print("----------------------------------")
        for name, value in config.items():
            if value:
                print(name, value)
    print("graph schedules:")
    for name, value in configs.graph_config.items():
        if value:
            print(name, value)
    string = json.dumps(configs)
    line = task.key + ":" + string
    print(line, file=logfile, flush=True)
    s, bufs = schedule_with_config(task.key, configs)
    time_cost = _evaluate(s, bufs, target, dev_id, 10)
    print("Use", time_cost, "ms", "throughput: %f GFLOPS" % (N * C * H * W * K * k * k / st / st / group / 1e6 / time_cost))
    print("Cost", end - beg, "s")

    logfile.close()
