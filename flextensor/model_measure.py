import tvm
from tvm.autotvm.measure.measure import Builder, Runner, MeasureResult, MeasureErrorNo
from flextensor.ppa_model import measure_latency
from flextensor.utils import get_iter_info
import time


class ModelBuilder(Builder):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def build(self, measure_inputs):
        build_results = []
        for target, task, config in measure_inputs:
            with target:
                try:
                    s, bufs = task.instantiate(config)
                    tvm.lower(s, bufs)
                    build_results.append(get_iter_info(s))
                except Exception as e:
                    print(e)
                    build_results.append(None)
        return build_results


class ModelRunner(Runner):
    def __init__(self, *args, **kwargs):
        super(ModelRunner,  self).__init__(*args, **kwargs)

    def get_build_kwargs(self):
        return {}

    def run(self, measure_inputs, build_results):
        results = []
        for info in build_results:
            l = measure_latency(info)
            if l is None:
                results.append(MeasureResult(
                    ['inf'], MeasureErrorNo.RUNTIME_DEVICE, 'inf', time.time()))
            else:
                results.append(MeasureResult(
                    [float(l)], MeasureErrorNo.NO_ERROR, float(l), time.time()))

        return results
