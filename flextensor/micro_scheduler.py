from flextensor.scheduler import schedule
from flextensor.task import register_task
from flextensor.utils import get_iter_info, RpcInfo
from flextensor.ppa_model import measure_latency
from flextensor.intrinsic import register_intrin


def gen_micro_schedule(task, intrin, model_measurer=measure_latency):
    register_task(task, override=True)
    register_intrin(intrin, override=True)
    rpc_info = RpcInfo(None, None)
    rpc_info.target = intrin.target
    s, _, _ = schedule(
        task.key,
        slevel=2, rlevel=2,
        model_measurer=model_measurer,
        rpc_info=rpc_info
    )
    return get_iter_info(s)
