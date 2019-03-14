import tvm
import logging
import sys
import time
from tvm import autotvm
from auto_schedule.scheduler import parallel_evaluate


@autotvm.template
def conv2d_channel_batch(B, N, M, C, K, L, O, stride=1, padding=0, dtype="float32"):
    A = tvm.placeholder((B, N, M, C), dtype=dtype, name="A")
    W = tvm.placeholder((K, L, C, O), dtype=dtype, name="W")
    N_out = max(0, (N + padding * 2 - K) // stride) + 1
    M_out = max(0, (M + padding * 2 - L) // stride) + 1
    Apad = tvm.compute((B, N + 2 * padding, M + 2 * padding, C),
                       lambda b, i, j, k: tvm.if_then_else(
                           tvm.all(i >= padding, j >= padding, i < N + padding, j < M + padding),
                           A[b, i - padding, j - padding, k], 0.0), name="Apad")
    rx, ry = tvm.reduce_axis((0, K), name="rx"), tvm.reduce_axis((0, L), name="ry")
    rc = tvm.reduce_axis((0, C), name="rc")
    Output = tvm.compute((B, N_out, M_out, O),
                         lambda b, i, j, k: tvm.sum(Apad[b, i * stride + rx, j * stride + ry, rc] * W[rx, ry, rc, k],
                                                    axis=[rx, ry, rc]),
                         name="Output")

    s = tvm.create_schedule(Output.op)
    s[Apad].compute_inline()
    CL = s.cache_write(Output, "local")

    n, h, w, c = s[Output].op.axis
    out = s[Output].fuse(h, w)
    cfg = autotvm.get_config()
    cfg.define_split("split_n", n, num_outputs=2)
    cfg.define_split("split_c", c, num_outputs=2)
    no, ni = cfg["split_n"].apply(s, Output, n)
    co, ci = cfg["split_c"].apply(s, Output, c)
    s[Output].reorder(no, out, co, ni, ci)
    s[Output].parallel(out)

    # schedule CL
    s[CL].compute_at(s[Output], co)
    ni, hi, wi, ci = s[CL].op.axis
    xi, yi, ki = s[CL].op.reduce_axis
    cfg.define_split("split_k", ki, num_outputs=2)
    ko, ki = cfg["split_k"].apply(s, CL, ki)
    s[CL].reorder(ko, xi, yi, ni, ki, ci)
    s[CL].unroll(ki)
    s[CL].vectorize(ci)

    return s, [A, W, Output]


args = (1, 14, 14, 256, 3, 3, 512, 1, 1)
task = autotvm.task.create(conv2d_channel_batch, args=args, target="llvm")

logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=10))

# begin tuning
tuner = autotvm.tuner.XGBTuner(task)
# tuner = autotvm.tuner.RandomTuner(task)
# tuner = autotvm.tuner.GATuner(task)
# tuner = autotvm.tuner.GridSearchTuner(task)
n_trial = len(task.config_space)
print("trials=", n_trial)
beg = time.time()
tuner.tune(n_trial=n_trial, measure_option=measure_option, callbacks=[autotvm.callback.log_to_file("conv2d.log")])
end = time.time()
# history best
with autotvm.apply_history_best("conv2d.log"):
    with tvm.target.create("llvm"):
        s, bufs = conv2d_channel_batch(*args)
        func = tvm.build(s, bufs)

# time evaluate
time_cost = parallel_evaluate(s, bufs, "llvm", 2)
print("time cost is: ", time_cost, "ms, use ",(end - beg), "s")
