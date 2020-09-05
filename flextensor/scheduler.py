import os
import time
import shutil
import math
import tvm
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from collections import deque
from flextensor.task import TASK_TABLE

try:
    from flextensor.model import WalkerGroup
except ImportError:
    print("[Warning] Import model module failed, please check if PyTorch is installed.")
from flextensor.space import generate_space_inter_op, generate_space_intra_op
from flextensor.utils import assert_print, to_tuple, Config, RpcInfo
from flextensor.templates import _cpu_schedule_simple, _cuda_schedule_split_reorder_fuse, \
    _opencl_schedule
from flextensor.measure import build_and_eval, master_routine, mp
from functools import partial
from tempfile import mkstemp, mkdtemp

try:
    import psutil
except ImportError:
    raise RuntimeError("psutil not found, please install it [Hint: `pip install psutil`]")


def flatten_graph(ops):
    bfs_order = []
    down_graph = {}
    visited = set()
    q = deque()
    for op in ops:
        q.append(op)
        visited.add(op)
    while q:
        cur = q.popleft()
        if isinstance(cur, tvm.te.tensor.ComputeOp):
            bfs_order.append(cur)
        for t in cur.input_tensors:
            if t.op not in visited:
                visited.add(t.op)
                q.append(t.op)
            if t not in down_graph:
                down_graph[t] = []
            down_graph[t].append(cur)
    return list(reversed(bfs_order)), down_graph


def find_idle_cpu():
    return 0


def find_idle_gpu():
    return 0


def find_idle_device(target):
    if target == "llvm":
        return find_idle_cpu()
    elif target == "cuda":
        return find_idle_gpu()
    else:
        raise RuntimeError("Currently no support for target %s" % target)


def build_and_eval_wrapper(lib, task_key, config, op_pos=None, rpc_info=None, rewrite=False,
                           number=1):
    task = TASK_TABLE[task_key]
    s, bufs = schedule_with_config(task_key, config, op_pos=op_pos, rewrite=rewrite)
    return build_and_eval(lib, s, bufs, task.target, task.dev_id, rpc_info, number)


class Scheduler(object):
    def __init__(self, name, task_key, space, parallel=2, timeout=4.0, trial=100, number=1,
                 early_stop=30,
                 rpc_info=None, rewrite=False):
        self.task_key = task_key
        self.space = space
        self.parallel = max(parallel, 1)  # at least 1
        self.timeout = timeout
        self.trial = trial
        self.number = number
        self.early_stop = early_stop
        self.task = TASK_TABLE[self.task_key]
        self.walker_group = WalkerGroup(self.task.category + "_" + name, self.space)
        self.rpc_info = rpc_info
        self.rewrite = rewrite

        self.re_evalutate_number = 10
        self.warm_up_epoch = 20
        self.warm_up_number = 20

        self._pool = None

    def _warm_up(self, warm_up_epoches, warm_up_trials, configs, type_keys, max_repeat=20,
                 use_model=False):
        # perform warmup
        warm_up_enough = False
        count_repeat = 0
        old_timeout = self.timeout
        while not warm_up_enough:
            for ep in range(warm_up_epoches):
                warm_up_ret = self.walker_group.forward(warm_up_trials, policy="random")
                warm_up_configs = [{} for _ in range(warm_up_trials)]  # empty configs
                warm_up_indices = [{} for _ in range(warm_up_trials)]  # the indices
                for count in range(warm_up_trials):
                    config = warm_up_configs[count]
                    for type_key in type_keys:
                        config[type_key] = []
                        for name in self.space.types[type_key]:
                            entity = warm_up_ret[name][0][count]
                            warm_up_indices[count][name] = warm_up_ret[name][1][count]
                            config[type_key].append(entity)
                    # hack here
                    # if self.op_pos == 1:
                    #     warm_up_configs[count] = {
                    #         "spatial": [[1, 1, 1, 1], [64, 2, 8, 1], [1, 1, 7, 1], [1, 1, 7, 1]],
                    #         "reduce": [[64, 1, 16], [1, 3, 1], [1, 1, 3]],
                    #         "unroll": [[1500, 1]]
                    #     }
                    # hack here
                    # warm_up_configs[count] = {"inline": [[False, False]]}
                if use_model:
                    warm_up_results = self.walker_group.query_performance(warm_up_indices)
                else:
                    warm_up_results = self.parallel_evaluate(configs, warm_up_configs,
                                                             number=self.number)
                    # the results are really measured
                    self.walker_group.add_perf_data(warm_up_indices, warm_up_results)
                string = "[ "
                for res in warm_up_results:
                    string += "%.6f " % res
                string += "]"
                print("warm up [%.6f] %s" % (time.time(), string))
                for count in range(warm_up_trials):
                    if warm_up_results[count] < float("inf"):
                        self.walker_group.record(warm_up_indices[count], warm_up_results[count])
                        # if not found valid config
            if not self.walker_group.top1():
                print("Warning: No valid schedule found in warm up process, please use more trials")
                print("Now automatically use more trials, increase %d" % warm_up_trials)
                warm_up_epoches = 1
                count_repeat += 1
                self.timeout = min(2 * self.timeout, 40)
                if count_repeat >= max_repeat:
                    print("Fail to find valid schedule, too many errors")
                    warm_up_enough = True
            else:
                warm_up_enough = True
        self.timeout = old_timeout

    def _random_schedule(self, configs, type_keys, use_model=False):
        # prepare model
        if use_model:
            self.walker_group.load_or_create_model()
        # random by warm-up
        for trial in range(self.trial):
            warm_up_epoches = 1
            warm_up_trials = self.parallel
            self._warm_up(warm_up_epoches, warm_up_trials, configs, type_keys, use_model=use_model)
        return self.walker_group.to_config(self.walker_group.top1())

    def _searching_schedule(self, configs, type_keys, use_model=False):
        # prepare model
        if use_model:
            self.walker_group.load_or_create_model()
        # warm up
        warm_up_epoches = self.warm_up_number
        warm_up_trials = self.warm_up_number
        self._warm_up(warm_up_epoches, warm_up_trials, configs, type_keys, use_model=use_model)

        # tune
        minimal = [{}, float("inf")]  # the minimal point found before
        retired_indices = []  # list of local minimals

        part = math.ceil(self.trial / 20)
        value_early_stop = self.walker_group.top1_value()
        early_stop_count = 0
        count_incessant_empty_trial = 0
        for trial in range(self.trial):
            if not self.walker_group.has_more():
                # nothing to tune, re-warm up
                warm_up_epoches = 1
                warm_up_trials = self.parallel
                self._warm_up(warm_up_epoches, warm_up_trials, configs, type_keys,
                              use_model=use_model)
                continue
            from_indices, from_value = self.walker_group.top_random(with_value=True)
            # # print("check from", from_indices)
            # get all directions
            next_indices_lst, action_lst = self.walker_group.full_walk(from_indices, no_repeat=True)
            # # print("check action", action_lst)
            next_configs = [self.walker_group.to_config(indices) for indices in next_indices_lst]
            # if empty
            if len(next_configs) < 1:
                count_incessant_empty_trial += 1
            else:
                count_incessant_empty_trial = 0
            if use_model:
                results = self.walker_group.query_performance(next_indices_lst)
            else:
                results = self.parallel_evaluate(configs, next_configs, number=self.number)
                # the results are really measured
                self.walker_group.add_perf_data(next_indices_lst, results)
            string = "[ "
            for res in results:
                string += "%.6f " % res
            string += "]"
            print("tune [%.6f] %s" % (time.time(), string))
            rewards = [np.tanh(max(from_value - result, 0.0)) for result in results]

            is_local_minimal = True
            for indices, action, reward, result in zip(next_indices_lst, action_lst, rewards,
                                                       results):
                self.walker_group.add_data(
                    action[0],  # name
                    from_indices,  # pre_state
                    action[1],  # action
                    indices,  # post_state
                    reward  # reward
                )
                self.walker_group.record(indices, result, random_reject=True)
                if result < self.walker_group.top1_value():
                    is_local_minimal = False
            # for local minimal value, remove OR no more exploration, remove
            if is_local_minimal or count_incessant_empty_trial > 0:
                top = self.walker_group.pop_top()
                if top.value < minimal[1]:
                    if minimal[1] < float("inf"):
                        retired_indices.append(minimal)
                    minimal[1] = top.value
                    minimal[0] = top.indices
                else:
                    retired_indices.append([top.indices, top.value])
            # report best
            if self.walker_group.top1_value() < minimal[1]:
                cur_best_value = self.walker_group.top1_value()
                cur_best = self.walker_group.top1()
            else:
                cur_best_value = minimal[1]
                cur_best = minimal[0]
            print("No. %d | [%.6f] The best currently %.6f" % (trial, time.time(), cur_best_value),
                  cur_best)
            # early stop becasue of lasting empty trials
            if count_incessant_empty_trial >= self.early_stop:
                print("Early stop after continuous no trials %d times" % (
                    count_incessant_empty_trial))
                break
            # early stop because of repeating value
            if math.fabs(cur_best_value - value_early_stop) < 0.02:
                early_stop_count += 1
            else:
                value_early_stop = cur_best_value
                early_stop_count = 0
            if early_stop_count >= self.early_stop:
                print("Early stop with value %f repeats %d times" % (
                    value_early_stop, early_stop_count))
                break
                # train and re-evaluate
            if (trial + 1) % part == 0:
                if not use_model:
                    # re-evaluate
                    if minimal[1] < float("inf"):
                        self.walker_group.record(minimal[0], minimal[1], random_reject=False)
                    for retired in retired_indices:
                        self.walker_group.record(retired[0], retired[1], random_reject=False)
                    minimal[0] = {}
                    minimal[1] = float("inf")

                    indices_lst = self.walker_group.topk(self.re_evalutate_number, modify=True)
                    next_configs = [self.walker_group.to_config(indices) for indices in indices_lst]
                    # use serialized evaluation
                    old_parallel = self.parallel
                    if self.task.target == "cuda":
                        self.parallel = 1
                    else:
                        self.parallel = 1  # min(self.parallel, os.cpu_count())
                    results = self.parallel_evaluate(configs, next_configs, number=self.number)
                    # recover parallel number
                    self.parallel = old_parallel
                    self.walker_group.add_perf_data(indices_lst, results)
                    string = "[ "
                    for res in results:
                        string += "%.6f " % res
                    string += "]"
                    print("re-evaluate [%.6f] %s" % (time.time(), string))
                    for indices, result in zip(indices_lst, results):
                        if result < float("inf"):
                            # if inf, maybe this measure is wrong
                            self.walker_group.record(indices, result, random_reject=False)
                # dump data
                # self.walker_group.dump_data()
                self.walker_group.clear_data()
            # re-warm up
            warm_up_epoches = 1
            warm_up_trials = self.parallel
            self._warm_up(warm_up_epoches, warm_up_trials, configs, type_keys, use_model=use_model)
        # the best
        if self.walker_group.top1_value() < minimal[1]:
            best = self.walker_group.top1()
        else:
            best = minimal[0]
        return self.walker_group.to_config(best)

    def _q_schedule(self, configs, type_keys, use_model=False):
        # prepare model
        self.walker_group.load_walker_model()
        if use_model:
            self.walker_group.load_or_create_model()
        # warm up
        warm_up_epoches = 10
        warm_up_trials = 20
        self._warm_up(warm_up_epoches, warm_up_trials, configs, type_keys, use_model=use_model)

        # record best
        best = self.walker_group.top1()
        best_value = self.walker_group.top1_value()
        retired_indices = []
        # early stop value
        value_early_stop = best_value
        early_stop_count = 0
        # determine start points
        cur_lst = self.walker_group.topk(self.parallel, modify=True, with_value=True)
        part = math.ceil(self.trial / 5)
        for trial in range(self.trial):
            from_lst, next_points, action_lst = self.walker_group.walk(cur_lst, trial)
            if use_model:
                results = self.walker_group.query_performance(next_points)
            else:
                next_configs = [self.walker_group.to_config(indices) for indices in next_points]
                results = self.parallel_evaluate(configs, next_configs, number=self.number)
                self.walker_group.add_perf_data(next_points, results)
            for indices, action, (from_indices, from_value), result in zip(next_points, action_lst,
                                                                           from_lst, results):
                reward = np.tanh(max(from_value - result, 0.0))
                self.walker_group.add_data(
                    action[0],  # name
                    from_indices,  # pre_state
                    action[1],  # action
                    indices,  # post_state
                    reward  # reward
                )
                self.walker_group.record(indices, result, random_reject=True)
            # update best
            if self.walker_group.top1_value() < best_value:
                best_value = self.walker_group.top1_value()
                best = self.walker_group.top1()
            print("No. %d | [%.6f] The best currently %.6f" % (trial, time.time(), best_value),
                  best)
            # early stop
            if math.fabs(best_value - value_early_stop) < 0.02:
                early_stop_count += 1
            else:
                value_early_stop = best_value
                early_stop_count = 0
            if early_stop_count >= self.early_stop:
                print("Early stop with value %f repeats %d times" % (
                    value_early_stop, early_stop_count))
                break
                # empty, stop
            if not self.walker_group.has_more():
                print("No more points, end of scheduling")
                break
            # reload next points
            retired_indices.extend(cur_lst)
            cur_lst = self.walker_group.topk(self.parallel, modify=True, with_value=True)
            if (trial + 1) % part == 0:
                self.walker_group.train_walkers()
                if not use_model:
                    # re-evaluate
                    if best_value < float("inf"):
                        self.walker_group.record(best, best_value, random_reject=False)
                        best = {}
                        best_value = float("inf")
                    for indices, value in retired_indices[-self.parallel:-1]:
                        self.walker_group.record(indices, value, random_reject=False)
                    indices_lst = self.walker_group.topk(self.parallel, modify=True)
                    print("check next indices:", indices_lst)
                    next_configs = [self.walker_group.to_config(indices) for indices in indices_lst]
                    results = self.parallel_evaluate(configs, next_configs, number=self.number)
                    self.walker_group.add_perf_data(indices_lst, results)
                    string = "[ "
                    for res in results:
                        string += "%.6f " % res
                    string += "]"
                    print("re-evaluate [%.6f] %s" % (time.time(), string))
                    for indices, result in zip(indices_lst, results):
                        self.walker_group.record(indices, result, random_reject=False)
                # re-warm up
                warm_up_epoches = 1
                warm_up_trials = self.parallel
                self._warm_up(warm_up_epoches, warm_up_trials, configs, type_keys,
                              use_model=use_model)
                # update best
                if self.walker_group.top1_value() < best_value:
                    best_value = self.walker_group.top1_value()
                    best = self.walker_group.top1()
        # dump data at last
        # self.walker_group.dump_data()
        self.walker_group.clear_data()
        return self.walker_group.to_config(best)

    def parallel_evaluate(self, old_configs, new_configs, number=1):
        raise NotImplementedError()

    def _parallel_evaluate(self, old_configs, new_configs, mode="op", number=1):
        # print("check config", old_configs, new_configs)

        res_lst = []
        lib_dir = mkdtemp(prefix="flextensor_lib_")
        task_key = self.task_key
        rpc_info = self.rpc_info
        rewrite = self.rewrite

        def _fetch_config(self, config):
            if mode == "op":
                build_config = Config(old_configs.op_config_lst + [config],
                                      old_configs.graph_config)
                op_pos = self.op_pos
            elif mode == "graph":
                build_config = Config(old_configs.op_config_lst, config)
                op_pos = None
            else:
                raise RuntimeError("Unknown mode %s" % mode)
            return build_config, op_pos

        def _serial_eval():
            for config in new_configs:
                build_config, op_pos = _fetch_config(self, config)
                fd, lib = mkstemp(prefix="flextensor_builtin_function_", suffix=".so",
                                  dir=lib_dir)
                os.close(fd)
                try:
                    res = build_and_eval_wrapper(lib, task_key, build_config, op_pos,
                                                 rpc_info, rewrite, number)
                except Exception as e:
                    res = float("inf")
                res_lst.append(res)

        def _par_eval():
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.parallel,
                    mp_context=mp.get_context())
            fut_lst = []
            for config in new_configs:
                build_config, op_pos = _fetch_config(self, config)
                fd, lib = mkstemp(prefix="flextensor_builtin_function_", suffix=".so",
                                  dir=lib_dir)
                os.close(fd)
                fut_lst.append(self._pool.submit(
                    master_routine,
                    self.timeout,
                    build_and_eval_wrapper,
                    lib, self.task_key, build_config,
                    op_pos, self.rpc_info, self.rewrite, number
                ))
            broken = False
            for fut in fut_lst:
                try:
                    res = fut.result()
                    if isinstance(res, Exception):
                        res = float("inf")
                except Exception as e:
                    broken = True
                    res = float("inf")
                res_lst.append(res)
            if broken:
                del self._pool
                self._pool = None

        try:
            if self.parallel == 1:
                _serial_eval()
            else:
                _par_eval()
        finally:
            shutil.rmtree(lib_dir)
        return res_lst


class OpScheduler(Scheduler):
    def __init__(self, task_key, op_pos, space, decay=0.7, parallel=1, timeout=4.0, trial=100,
                 number=1, early_stop=30,
                 rpc_info=None, rewrite=False):
        super(OpScheduler, self).__init__("op" + str(op_pos), task_key, space, parallel, timeout,
                                          trial, number,
                                          early_stop, rpc_info, rewrite=rewrite)
        self.op_pos = op_pos

    def schedule(self, configs, method="searching", use_model=False, perf_path=None):
        # if hint == "split_fuse":
        #     wanted_types = ["spatial", "reduce", "unroll"]
        # elif hint == "fuse_split":
        #     wanted_types = ["fuse", "reorder", "spatial", "reduce", "unroll"]
        # else:
        #     raise RuntimeError("Unknown hint: %s" % hint)
        wanted_types = ["fuse", "reorder", "spatial", "reduce", "unroll"]
        if perf_path is not None:
            self.walker_group.model_path = perf_path
        if method == "searching":
            return self._searching_schedule(configs, wanted_types, use_model=use_model)
        elif method == "q":
            return self._q_schedule(configs, wanted_types, use_model=use_model)
        elif method == "random":
            return self._random_schedule(configs, wanted_types, use_model=use_model)
        else:
            raise RuntimeError("Currently no support for method %s" % method)

    def parallel_evaluate(self, configs, next_op_configs, number=1, rpc_info=None):
        return self._parallel_evaluate(configs, next_op_configs, mode="op", number=number)

    @staticmethod
    def generate_op_schedule(target, config):
        template = None
        if target == "cuda":
            # if hint == "split_fuse":
            #     print(hint)
            #     return _cuda_schedule_split_fuse
            # elif hint == "fuse_split":
            #     print(hint)
            #     return _cuda_schedule_fuse_split
            # else:
            #     raise RuntimeError("Unknown hint: %s" % hint)
            template = _cuda_schedule_split_reorder_fuse
        elif target == "llvm":
            template = _cpu_schedule_simple
        elif target == "opencl":
            template = _opencl_schedule
        elif target[0] == "c":
            # this is for c code generation
            dev_keys = target.split()
            if "-device=micro_dev" in dev_keys:
                pass
        else:
            raise RuntimeError("Currently no support for target %s" % target)
        return partial(template, config)


class Rewriter(object):
    def __init__(self, configs):
        self.graph_config = configs.graph_config
        self.op_config_lst = configs.op_config_lst

    def rewrite(self, task):
        """
        this is a hard code manner,
        we don't know how to generalize this change
        because it even need compute rewrite and schedule rewrite
        """
        assert task.target == "llvm", "Only rewrite for CPU"
        assert task.category == "conv2d"
        # schedule rewrite
        import copy
        new_graph_config = copy.deepcopy(self.graph_config)
        new_op_config_lst = copy.deepcopy(self.op_config_lst)
        # must compute inline as original config may split channel differently
        new_graph_config["inline"] = [[1, 0]]
        # fetch conv config
        conv_config = self.op_config_lst[1]
        new_config = new_op_config_lst[1]
        # change out_channel config
        vlen1 = conv_config["reduce"][0][-1]
        vlen2 = conv_config["spatial"][1][-1]
        new_config["spatial"].append([1] * len(new_config["spatial"][0]))
        new_config["spatial"][-1][-1] = vlen2
        new_config["spatial"][1][-1] = 1
        new_config["reduce"][0][-1] = 1
        new_config["reduce"].insert(1, [1] * len(new_config["reduce"][0]))
        new_config["reduce"][1][-1] = vlen1
        # compute rewrite
        from flextensor.task import conv2d_nchwc_layout
        kwargs = {"vlen1": vlen1, "vlen2": vlen2}
        ops, bufs = conv2d_nchwc_layout(*task.args, **kwargs)
        return ops, bufs, new_graph_config, new_op_config_lst


class GraphScheduler(Scheduler):
    def __init__(self, task_key, space, decay=0.7, parallel=10, timeout=4.0, trial=100, number=1,
                 early_stop=30,
                 rpc_info=None, rewrite=False):
        super(GraphScheduler, self).__init__("graph", task_key, space, parallel, timeout, trial,
                                             number, early_stop,
                                             rpc_info, rewrite=rewrite)

    def schedule(self, configs, method="searching", use_model=False, perf_path=None):
        if perf_path is not None:
            self.walker_group.model_path = perf_path
        if method == "searching":
            return self._searching_schedule(configs, ["inline", "merge"], use_model=use_model)
        elif method == "q":
            return self._q_schedule(configs, ["inline", "merge"], use_model=use_model)
        elif method == "random":
            return self._random_schedule(configs, ["inline", "merge"], use_model=use_model)
        else:
            raise RuntimeError("Currently no support for method %s" % method)

    def parallel_evaluate(self, configs, graph_configs, number=1):
        return self._parallel_evaluate(configs, graph_configs, mode="graph", number=number)

    @staticmethod
    def generate_graph_schedule(config, phase="inline"):
        def _inline_schedule(s, op_lst, op_states):
            if "inline" in config and len(config["inline"]) > 0:
                entity = config["inline"][0]
                for count in range(len(op_lst)):
                    if entity[count]:
                        s[op_lst[count]].compute_inline()
                        op_states[count].inline = True

        def _at_schedule(s, op_lst, op_states):
            return
            if "merge" in config and len(config["merge"]) > 0:
                entity = config["merge"][0]
                for count in range(len(op_lst)):
                    if entity[count] >= 0:
                        num_consumers = len(op_states[count].consumer_lst)
                        if num_consumers != 1 or op_states[count].inline:
                            continue
                        else:
                            consumer_id = op_states[count].consumer_lst[0]
                            consumer_state = op_states[consumer_id]
                            if consumer_state.inline:
                                continue  # do not compute at inlined ops
                            consumer_loop_idx = consumer_state.loop_idx
                            at_pos = consumer_state.loop_lst[consumer_loop_idx[entity[count]]]
                            s[op_lst[count]].compute_at(s[op_lst[consumer_id]], at_pos)
                            op_states[count].compute_at = True

        if phase == "inline":
            return _inline_schedule
        elif phase == "at":
            return _at_schedule
        else:
            raise RuntimeError("Currently no support for phase %s" % phase)


class OpState(object):
    def __init__(self):
        self.inline = False
        self.loop_lst = []
        self.loop_idx = []
        self.compute_at = False
        self.consumer_lst = []


def schedule(task_key, slevel=4, rlevel=3, op_trial=50, graph_trial=10, op_stop=15, graph_stop=5,
             number=1, timeout=5.0, parallel=8, method="searching", **kwargs):
    """Schedule a task

    perform sequential schedule
    """
    task = TASK_TABLE[task_key]
    func = task.func
    args = task.args
    ops, bufs = func(*args)
    # sort the ops, so that we can distinguish each op
    op_lst, down_graph = flatten_graph(ops)
    # state of ops
    op_states = [OpState() for op in op_lst]
    for count_op, op in enumerate(op_lst):
        consumer_lst = []
        for count_output in range(op.num_outputs):
            if op.output(count_output) in down_graph:
                consumer_lst.extend(down_graph[op.output(count_output)])
        op_states[count_op].consumer_lst = list(set(consumer_lst))

    if "trials" in kwargs:
        assert_print(len(kwargs["trials"]) == len(op_lst), str(len(op_lst)))
        force_trials = kwargs["trials"]
    else:
        force_trials = [op_trial for i in range(len(op_lst))]

    op_perf_model_path_lst = [None for i in range(len(op_lst))]
    if "op_perf_model_path" in kwargs:
        for (op_pos, path) in kwargs["op_perf_model_path"]:
            op_perf_model_path_lst[op_pos] = path
    graph_perf_model_path = None
    if "graph_perf_model_path" in kwargs:
        graph_perf_model_path = kwargs["graph_perf_model_path"]
    force_inline = False
    if "force_inline" in kwargs:
        force_inline = kwargs["force_inline"]
    if "rewrite" in kwargs:
        rewrite = True
        # must force_inline
        force_inline = True
    else:
        rewrite = False
    rpc_info = None
    if "rpc_info" in kwargs:
        rpc_info = kwargs["rpc_info"]
    ##################################################
    # first generate graph space
    graph_space = generate_space_inter_op(op_lst, down_graph, force_inline=force_inline,
                                          special_space=task.special_space)

    graph_space_size = len(graph_space)
    print("graph space size", graph_space_size)
    total_size = graph_space_size

    ##################################################
    # intra operations schedule decisionss
    op_space_lst = []
    if force_inline:
        configs = Config([], {"inline": [graph_space.subspaces["inline"].static_entities[0]]})
    else:
        configs = Config([], None)

    for pos, op in enumerate(op_lst):
        if task.target == "cuda" or task.target == "opencl":
            space = generate_space_intra_op(op, down_graph, slevel=slevel, rlevel=rlevel, groups=3)
        elif task.target == "llvm":
            rslevel = max(slevel, rlevel)
            space = generate_space_intra_op(op, down_graph, slevel=rslevel, rlevel=rslevel,
                                            unroll_policy="off", fuse_policy="off",
                                            reorder_policy="off")
        else:
            raise RuntimeError("Currently no support for target %s" % task.target)
        total_size *= len(space)
        print("op", pos, "space size:", len(space))
        op_space_lst.append(space)
        op_scheduler = OpScheduler(
            task_key,
            pos,
            space,
            parallel=parallel,
            timeout=timeout,
            trial=force_trials[pos],
            number=number,
            early_stop=op_stop,
            rpc_info=rpc_info,
            rewrite=rewrite
        )
        # print("###########################################")
        # print("Scheduling", op)
        use_model = False if op_perf_model_path_lst[pos] is None else True
        perf_path = op_perf_model_path_lst[pos]
        if force_inline and graph_space.subspaces["inline"].able_inline(pos):
            op_config = {}
        else:
            op_config = op_scheduler.schedule(
                configs,
                method=method,
                use_model=use_model,
                perf_path=perf_path,
            )
        configs.op_config_lst.append(op_config)

    print("space size", total_size)

    #################################################
    # inter operations schedule decisions 
    graph_scheduler = GraphScheduler(
        task_key,
        graph_space,
        parallel=parallel,
        timeout=timeout,
        trial=graph_trial,
        number=number,
        early_stop=graph_stop,
        rpc_info=rpc_info,
        rewrite=rewrite
    )
    use_model = False if graph_perf_model_path is None else True
    graph_config = graph_scheduler.schedule(configs, method=method, use_model=use_model,
                                            perf_path=graph_perf_model_path)
    #################################################
    # combine the configs
    configs = Config(configs.op_config_lst, graph_config)

    #################################################
    # final schedule
    # s = tvm.te.create_schedule(ops)
    # # perform inter operator schedule
    # graph_template = GraphScheduler.generate_graph_schedule(configs.graph_config, phase="inline")
    # graph_template(s, op_lst, op_states)
    # # perform intra-operator schedule
    # for count_op, (op, op_state, op_config) in enumerate(zip(op_lst, op_states, configs.op_config_lst)):
    #     if not op_state.inline:
    #         op_template = OpScheduler.generate_op_schedule(task.target, op_config)
    #         op_template(s, op, op_states[count_op])
    # # perform inter operations schedule again for compute at
    # if graph_config is not None:
    #     graph_template = GraphScheduler.generate_graph_schedule(graph_config, phase="at")
    #     graph_template(s, op_lst, op_states)
    s, bufs = schedule_with_config(task_key, configs, rewrite=rewrite)

    return s, bufs, configs


def schedule_with_config(task_key, configs, op_pos=None, rewrite=False):
    """Schedule a task with given configs

    perform sequential schedule
    """
    task = TASK_TABLE[task_key]
    rewriter = Rewriter(configs)
    if rewrite:
        ops, bufs, new_graph_config, new_op_config_lst = rewriter.rewrite(task)
        configs = Config(new_op_config_lst, new_graph_config)
    else:
        func = task.func
        args = task.args
        ops, bufs = func(*args)

    s, bufs = schedule_with_config_ops(ops, bufs, configs, op_pos=op_pos, target=task.target)
    return s, bufs


def schedule_with_config_ops(ops, bufs, configs, op_pos=None, target="llvm"):
    """Schedule a task with given configs

    perform sequential schedule
    """
    # sort the ops, so that we can distinguish each op
    op_lst, down_graph = flatten_graph(ops)
    # state of ops
    op_states = [OpState() for op in op_lst]
    for count_op, op in enumerate(op_lst):
        consumer_lst = []
        for count_output in range(op.num_outputs):
            if op.output(count_output) in down_graph:
                consumer_lst.extend(down_graph[op.output(count_output)])
        op_states[count_op].consumer_lst = list(set(consumer_lst))

    op_config_lst = configs.op_config_lst

    if op_pos is not None:
        assert_print(isinstance(op_pos, int), "op_pos should be int")
        assert_print(op_pos < len(op_lst) and op_pos < len(op_config_lst), "op_pos too big")
        loop_length = op_pos + 1
        s = tvm.te.create_schedule(op_lst[op_pos])
    else:
        assert_print(len(op_config_lst) <= len(op_lst), "config length exceed op_lst")
        loop_length = len(op_config_lst)
        s = tvm.te.create_schedule(ops)

    ###################################################
    # perform inter operations schedule first for inline
    graph_config = configs.graph_config
    if graph_config is not None:
        graph_template = GraphScheduler.generate_graph_schedule(graph_config, phase="inline")
        graph_template(s, op_lst, op_states)

    ###################################################
    # perform intra operations schedule    
    for i in range(loop_length):
        # mask inlined ops
        if not op_states[i].inline:
            op = op_lst[i]
            config = op_config_lst[i]
            template = OpScheduler.generate_op_schedule(target, config)
            template(s, op, op_states[i])

            ###################################################
    # perform inter operations schedule again for compute at
    if graph_config is not None:
        graph_template = GraphScheduler.generate_graph_schedule(graph_config, phase="at")
        graph_template(s, op_lst, op_states)

    return s, bufs


def schedule_ops_with_config(s, op_lst, configs, target):
    """
    Schedule op list with given configs
    This assumes a previous graph optimizaton
    so there is no need to retrieve graph list 
    nor perform compute_at
    """
    # state of ops
    op_states = [OpState() for op in op_lst]

    op_config_lst = configs.op_config_lst

    loop_length = len(op_config_lst)

    ###################################################
    # perform inter operations schedule first for inline
    graph_config = configs.graph_config
    if graph_config is not None:
        graph_template = GraphScheduler.generate_graph_schedule(graph_config, phase="inline")
        graph_template(s, op_lst, op_states)

    ###################################################
    # perform intra operations schedule    
    for i in range(loop_length):
        # mask inlined ops
        if not op_states[i].inline:
            op = op_lst[i]
            config = op_config_lst[i]
            template = OpScheduler.generate_op_schedule(target, config)
            template(s, op, op_states[i])

    return s
