import os
import time
import signal
import shutil
import math
import tvm
import numpy as np
import torch.multiprocessing as _multi
multi = _multi.get_context("fork")
from tvm import rpc
from collections import deque, namedtuple
from queue import Empty
from functools import reduce
from auto_schedule.testing.task import TASK_TABLE
from auto_schedule.testing.model import WalkerGroup
from auto_schedule.testing.space import generate_space_inter_op, generate_space_intra_op
from auto_schedule.utils import assert_print, to_tuple
try:
    import psutil
except ImportError:
    raise RuntimeError("psutil not found, please install it [Hint: `pip install psutil`]")


LIB_DIR = "lib"
LOCAL_RPC = False


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
        if isinstance(cur, tvm.tensor.ComputeOp):
            bfs_order.append(cur)
        for t in cur.input_tensors:
            if t.op not in visited:
                visited.add(t.op)
                q.append(t.op)
            if t not in down_graph:
                down_graph[t] = []
            down_graph[t].append(cur)
    return list(reversed(bfs_order)), down_graph

def verify_code(stmt, target, dev_id):
    if target == "cuda":
        ctx = tvm.nd.context(target, dev_id)     # just use device 0
        if not ctx.exist:
            # print("Fail to get device %s devid=%d"%(target, dev_id))
            return False
        max_dims = ctx.max_thread_dimensions
        check_gpu = {
            "max_shared_memory_per_block": ctx.max_shared_memory_per_block,
            "max_threads_per_block": ctx.max_threads_per_block,
            "max_thread_x": max_dims[0],
            "max_thread_y": max_dims[1],
            "max_thread_z": max_dims[2]
        }
        valid = tvm.ir_pass.VerifyGPUCode(stmt, check_gpu)
        return valid
    else: 
        # no barrier for other targets
        return True


def build_func(func_name, task_key, configs, op_pos=None, rpc_info=None):
    # print("check 3.3.1-build")
    if rpc_info is not None and rpc_info.target_host is not None:
        target_host = rpc_info.target_host
    else:
        target_host = None
    task = TASK_TABLE[task_key]
    s, bufs = schedule_with_config(task_key, configs, op_pos=op_pos)
    stmt = tvm.lower(s, bufs, simple_mode=True)
    valid = verify_code(stmt, task.target, task.dev_id)
    if not valid:
        raise RuntimeError("Invalid %s(%d) kernel"%(task.target, task.dev_id))
    if target_host is not None:
        func = tvm.build(s, bufs, target=task.target, target_host=target_host)
    else:
        func = tvm.build(s, bufs, target=task.target)
    func.export_library(os.path.join(LIB_DIR, func_name))
    result = ([to_tuple(x.shape) for x in bufs], bufs[0].dtype)
    # print("check 3.3.2-build")
    return result


def eval_func(func_file, bufs_shape, dtype, target, number=100, dev_id=0, rpc_info=None):
    # print("check 3.3.1")
    if rpc_info is not None:
        host = rpc_info.host
        port = rpc_info.port
    else:
        # local
        host = "0.0.0.0"
        port = 9090     # default port
    if host == "0.0.0.0":
        if LOCAL_RPC:
            use_rpc = True
        else:
            use_rpc = False
    else:
        use_rpc = True
    if use_rpc:
        remote = rpc.connect(host, port)
        ctx = remote.context(target, dev_id)
    else:
        ctx = tvm.context(target, dev_id)
    # print("check 3.3.2")
    tvm_arys = []
    for shape in bufs_shape:
        shape = to_tuple(shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    # print("check 3.3.3")
    try:
        if use_rpc:
            # print("check 3.3.4a")
            remote.upload(os.path.join(LIB_DIR, func_file))
            func = remote.load_module(func_file)
            # print("check 3.3.5a")
        else:
            # print("check 3.3.4b")
            func = tvm.module.load(os.path.join(LIB_DIR, func_file))
            # print("check 3.3.5b")
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
        # print("check 3.3.6")
        time_cost = evaluator(*tvm_arys).mean * 1e3
        # print("check 3.3.7")
    finally:
        # print("check 3.3.8")
        while len(tvm_arys) > 0:
            del tvm_arys[-1]
        # print("check 3.3.9")
    return time_cost


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """kill all child processes recursively"""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            return


def exec_func(func, queue, args, kwargs):
    try:
        # print("check 3.3.0.0.1")
        res = func(*args, **kwargs)
        # print("check 3.3.0.0.2")
    except Exception as e:
        # print("check 3.3.0.0.3")
        res = RuntimeError(str(e))
        # print("check 3.3.0.0.4")
    queue.put(res)
    # print("check 3.3.0.0.5")


def parallel_execute(func, timeout, *args, **kwargs):
    # print("check 3.1")
    q = multi.Queue()
    # print("check 3.2")
    p = multi.Process(
        target=call_with_timeout, 
        args=(func, q, timeout, args, kwargs))
    # print("check 3.3")
    p.start()
    # print("check 3.4")
    return Result(p, q)


def call_with_timeout(func, queue, timeout, args, kwargs):
    q = multi.Queue()
    p = multi.Process(target=exec_func, args=(func, q, args, kwargs))
    # print("check 3.3.0.1")        
    # print("check 3.3.0.2")
    p.start()
    # print("check 3.3.0.3")
    # beg = time.time()
    # while time.time() - beg < timeout:
    #     if p.is_alive():
    #         time.sleep(.1)
    #     else:
    #         break
    try:
        res = q.get(block=True, timeout=timeout)
    except Empty:
        res = multi.TimeoutError()
    except Exception as e:
        print("Exception in process {}: {}".format(os.getpid(), str(e)))
        res = e
    # p.join(timeout=timeout)

    # print("check 3.3.0.4")
    # # print("check queue", queue.empty())
    # queue.put(multi.TimeoutError()) 
    # print("check 3.3.0.5")
    # kill_child_processes(p.pid)
    # print("check 3.3.0.6")
    # p.terminate()
    # print("check 3.3.0.7")
    # p.join()   
    # print("check 3.3.0.8")
    kill_child_processes(p.pid)
    p.terminate()
    p.join()
    queue.put(res)


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
        raise RuntimeError("Currently no support for target %s"%target)


class Config(namedtuple("Config", ("op_config_lst", "graph_config"))):
    pass


class Scheduler(object):
    def __init__(self, name, task_key, space, parallel=2, timeout=4.0, trial=100, number=10, early_stop=30, rpc_info=None):
        self.task_key = task_key
        self.space = space
        self.parallel = max(parallel, 1)    # at least 1
        self.timeout = timeout
        self.trial = trial
        self.number = number
        self.early_stop = early_stop
        self.task = TASK_TABLE[self.task_key]
        self.walker_group = WalkerGroup(self.task.category + "_" + name, self.space)
        self.rpc_info = rpc_info

        self.re_evalutate_number = 10
        self.warm_up_epoch = 20
        self.warm_up_number = 20

    def _warm_up(self, warm_up_epoches, warm_up_trials, configs, type_keys, max_repeat=20, use_model=False):
        # perform warmup
        warm_up_enough = False
        count_repeat = 0
        old_timeout = self.timeout
        while not warm_up_enough:
            for ep in range(warm_up_epoches):
                warm_up_ret = self.walker_group.forward(warm_up_trials, policy="random")
                warm_up_configs = [{} for i in range(warm_up_trials)]   # empty configs
                warm_up_indices = [{} for i in range(warm_up_trials)]   # the indices
                for count in range(warm_up_trials):
                    config = warm_up_configs[count]
                    for type_key in type_keys:
                        config[type_key] = []
                        for name in self.space.types[type_key]:
                            entity = warm_up_ret[name][0][count]
                            warm_up_indices[count][name] =  warm_up_ret[name][1][count]
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
                    warm_up_results = self.parallel_evaluate(configs, warm_up_configs, number=self.number)
                    # the results are really measured
                    self.walker_group.add_perf_data(warm_up_indices, warm_up_results)
                print("warm up", warm_up_results)
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
        warm_up_epoches = min(self.warm_up_epoch, math.ceil(self.trial / self.warm_up_number))
        warm_up_trials = self.warm_up_number
        self._warm_up(warm_up_epoches, warm_up_trials, configs, type_keys, use_model=use_model)
        
        # tune
        minimal = [{}, float("inf")]    # the minimal point found before
        retired_indices = []            # list of local minimals

        part = math.ceil(self.trial / 20)
        value_early_stop = self.walker_group.top1_value()
        early_stop_count = 0
        count_incessant_empty_trial = 0
        for trial in range(self.trial):
            if not self.walker_group.has_more():
                # nothing to tune, re-warm up
                warm_up_epoches = 1
                warm_up_trials = self.parallel
                self._warm_up(warm_up_epoches, warm_up_trials, configs, type_keys, use_model=use_model)
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
            print("tune", results)
            rewards = [np.tanh(max(from_value - result, 0.0)) for result in results]

            is_local_minimal = True
            for indices, action, reward, result in zip(next_indices_lst, action_lst, rewards, results):
                self.walker_group.add_data(
                                        action[0],      # name
                                        from_indices,   # pre_state
                                        action[1],      # action
                                        indices,        # post_state 
                                        reward          # reward
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
            print("No. %d | The best currently" % trial, cur_best_value, cur_best)
            # early stop becasue of lasting empty trials
            if count_incessant_empty_trial >= self.early_stop:
                print("Early stop after continuous no trials %d times" % (count_incessant_empty_trial))
                break
            # early stop because of repeating value
            if math.fabs(cur_best_value - value_early_stop) < 0.02:
                early_stop_count += 1
            else:
                value_early_stop = cur_best_value
                early_stop_count = 0
            if early_stop_count >= self.early_stop:
                print("Early stop with value %f repeats %d times" % (value_early_stop, early_stop_count))
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
                        self.parallel = min(self.parallel, os.cpu_count())
                    results = self.parallel_evaluate(configs, next_configs, number=self.number)
                    # recover parallel number
                    self.parallel = old_parallel
                    self.walker_group.add_perf_data(indices_lst, results)
                    print("re-evaluate", results)
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
        warm_up_epoches = self.trial // self.parallel
        warm_up_trials = self.parallel
        self._warm_up(warm_up_epoches, warm_up_trials, configs, type_keys, use_model=use_model)

        # record best
        best = self.walker_group.top1()
        best_value = self.walker_group.top1_value()
        retired_indices = []
        # early stop value
        value_early_stop = best_value
        early_stop_count = 0
        # determine start points
        cur_lst = self.walker_group.topk(4, modify=True, with_value=True)
        part = math.ceil(self.trial / 20)
        for trial in range(self.trial):
            from_lst, next_points, action_lst = self.walker_group.walk(cur_lst, trial)
            if use_model:
                results = self.walker_group.query_performance(next_points)
            else:
                next_configs = [self.walker_group.to_config(indices) for indices in next_points]
                results = self.parallel_evaluate(configs, next_configs, number=self.number)
                self.walker_group.add_perf_data(next_points, results)
            for indices, action, (from_indices, from_value), result in zip(next_points, action_lst, from_lst, results):
                reward = np.tanh(max(from_value - result, 0.0))
                self.walker_group.add_data(
                                        action[0],      # name
                                        from_indices,   # pre_state
                                        action[1],      # action
                                        indices,        # post_state 
                                        reward          # reward
                                        )
                self.walker_group.record(indices, result, random_reject=True)
            # update best
            if self.walker_group.top1_value() < best_value:
                best_value = self.walker_group.top1_value()
                best = self.walker_group.top1()
            print("The best currently", best_value, best)
            # early stop
            if math.fabs(best_value - value_early_stop) < 0.02:
                early_stop_count += 1
            else:
                value_early_stop = best_value
                early_stop_count = 0
            if early_stop_count >= self.early_stop:
                print("Early stop with value %f repeats %d times" % (value_early_stop, early_stop_count))
                break 
            # empty, stop
            if not self.walker_group.has_more():
                print("No more points, end of scheduling")
                break
            # reload next points
            retired_indices.extend(cur_lst)
            cur_lst = self.walker_group.topk(4, modify=True, with_value=True)    
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
                    next_configs = [self.walker_group.to_config(indices) for indices in indices_lst]
                    results = self.parallel_evaluate(configs, next_configs, number=self.number)
                    self.walker_group.add_perf_data(indices_lst, results)
                    print("re-evaluate", results)
                    for indices, result in zip(indices_lst, results):
                        self.walker_group.record(indices, result, random_reject=False)
                # re-warm up
                warm_up_epoches = 1
                warm_up_trials = self.parallel
                self._warm_up(warm_up_epoches, warm_up_trials, configs, type_keys, use_model=use_model)
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
        # # print("check config", old_configs, new_configs)
        # print("parallel_evaluate begins...")
        target = self.task.target
        total_configs = len(new_configs)
        total_res_lst = []
        try:
            os.mkdir(LIB_DIR)
        except OSError as e:
            if os.path.exists(LIB_DIR) and os.path.isdir(LIB_DIR):
                print("[Warning] Directory %s is not empty, but reusing it" % LIB_DIR)
            else:
                print("[Error] Fail to create directory %s\nReason: %s" % (LIB_DIR, str(e)))
                exit(1)
        for ep in range(math.ceil(total_configs / self.parallel)):
            part_configs = new_configs[ep * self.parallel:(ep + 1) * self.parallel]
            build_res_lst = []
            func_name_lst = []
            for config in part_configs:
                func_name = "auto_schedule_built_function_{}_{}.tar".format(time.time(), np.random.randint(1000, 10000))
                func_name_lst.append(func_name)
                if mode == "op":
                    build_config = Config(old_configs.op_config_lst + [config], old_configs.graph_config)
                    op_pos = self.op_pos
                elif mode == "graph":
                    build_config = Config(old_configs.op_config_lst, config)
                    op_pos = None
                else:
                    raise RuntimeError("Unknown mode %s" % mode)
                res = parallel_execute(
                    build_func, 
                    self.timeout, 
                    func_name,
                    self.task_key, 
                    build_config, 
                    op_pos,
                    rpc_info=self.rpc_info,
                    )
                build_res_lst.append(res)

            # time.sleep(self.timeout)
            
            eval_res_lst = []
            for i, build_res in enumerate(build_res_lst):
                # print("build result get begins...")
                final_res = build_res.get(timeout=self.timeout)
                # print("build resutl get done.")
                func_name = func_name_lst[i]
                if isinstance(final_res, Exception):
                    # print("check 1")
                    msg = mode + " build fail:"
                    # print(final_res.__class__)
                    if isinstance(final_res, multi.TimeoutError):
                        msg = msg + "Timeout"
                    elif isinstance(final_res, tvm._ffi.base.TVMError):
                        msg = msg + " TVMError "
                    error_str = str(final_res)
                    found = False
                    for key_word in ["TVMError", "Error", "error", "Fail", "fail", "Invalid", "invalid"]:
                        if key_word in error_str:
                            msg = msg + error_str[error_str.index(key_word):1000]
                            found = True
                            break
                    if not found:
                        msg = msg + error_str
                    # print(msg)
                    eval_res_lst.append(float("inf"))
                    # print("check 2")
                else:
                    # print("check 3")
                    res = parallel_execute(
                        eval_func,
                        self.timeout,
                        func_name,
                        final_res[0],
                        final_res[1],
                        target,
                        number=number,
                        dev_id=self.task.dev_id,
                        rpc_info=self.rpc_info
                    )
                    # print("check 3.9")
                    eval_res_lst.append(res)
                    # print("check 4")

            # time.sleep(self.timeout)

            ret_lst = []
            for eval_res in eval_res_lst:
                if isinstance(eval_res, float):
                    ret_lst.append(eval_res)
                else:
                    # print(print("evluate result getting...")
                    final_res = eval_res.get(timeout=self.timeout)
                    # print("evlaute result get done.")
                    if isinstance(final_res, Exception):
                        msg = mode + " run fail:"
                        # print(final_res.__class__)
                        if isinstance(final_res, multi.TimeoutError):
                            msg = msg + " Timeout "
                        elif isinstance(final_res, tvm._ffi.base.TVMError):
                            msg = msg + " TVMError "
                        error_str = str(final_res)
                        found = False
                        for key_word in ["Error", "error", "Fail", "fail", "Invalid", "invalid"]:
                            if key_word in error_str:
                                msg = msg + error_str[error_str.index(key_word):1000]
                                found = True
                                break
                        if not found:
                            msg = msg + error_str
                        # print(msg)
                        ret_lst.append(float("inf"))
                    else:
                        ret_lst.append(final_res)

            total_res_lst.extend(ret_lst)

            for func_name in func_name_lst:
                try:
                    os.remove(os.path.join(LIB_DIR, func_name))
                except FileNotFoundError:
                    pass
                    # print("File not found when deleting")
        try:
            shutil.rmtree(LIB_DIR)
        except Exception as e:
            print(e)
        # print("parallel evaluate done.")
        return total_res_lst


class OpScheduler(Scheduler):
    def __init__(self, task_key, op_pos, space, decay=0.7, parallel=1, timeout=4.0, trial=100, number=10, early_stop=30, rpc_info=None):
        super(OpScheduler, self).__init__("op" + str(op_pos), task_key, space, parallel, timeout, trial, number, early_stop, rpc_info)
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
        def _cuda_schedule_split_fuse(s, op, op_state):
            # assert_print(op in s)

            # always cache write here
            if op.num_outputs > 1:
                raise RuntimeWarning("Too many outputs in one operation!")
            write_cache = s.cache_write(op.output(0), "local")

            # always cache read here
            read_cache_share_lst = []
            read_cache_local_lst = []
            for t in op.input_tensors:
                share = s.cache_read(t, "shared", [write_cache])
                read_cache_share_lst.append(share)
                local = s.cache_read(share, "local", [write_cache])
                read_cache_local_lst.append(local)

            # spatial split
            spatial_axes = s[op].op.axis
            splited_spatial_axes = []
            if "spatial" in config and len(config["spatial"]) > 0:
                # to align each axis
                assert_print(len(config["spatial"]) == len(spatial_axes), "align failed")
                for axis, nparts in zip(spatial_axes, config["spatial"]):
                    tmp_buffer = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[op].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_spatial_axes.append(tmp_buffer)
            else:
                for axis in spatial_axes:
                    splited_spatial_axes.append([axis])
            assert_print(len(splited_spatial_axes) > 0, "empty spatial axes")     # must be non-empty

            # always reorder and fuse here
            spatial_fuse_lsts = []
            spatial_fuse_extents = []
            reorder_lst = []
            fused_spatial_axes = []
            for count in range(len(splited_spatial_axes[0])):
                tmp_buffer = [x[count] for x in splited_spatial_axes]
                tmp_extent = reduce(lambda a, b: a * b, [x[count] for x in config["spatial"]])
                spatial_fuse_lsts.append(tmp_buffer)
                spatial_fuse_extents.append(tmp_extent)
                reorder_lst.extend(tmp_buffer)
            s[op].reorder(*reorder_lst)
            for fuse_lst in spatial_fuse_lsts:
                fused = s[op].fuse(*fuse_lst)
                fused_spatial_axes.append(fused)
            kernel_scope = fused_spatial_axes[0]
            
            # always bind here
            length = len(fused_spatial_axes)
            thread_extents = 1
            assert_print(length > 1, "fused axes length <= 1")
            if 2 <= length <= 3:
                s[op].bind(fused_spatial_axes[0], tvm.thread_axis("blockIdx.x"))
                s[op].bind(fused_spatial_axes[1], tvm.thread_axis("threadIdx.x"))
                thread_pos = fused_spatial_axes[1]
                thread_extents = spatial_fuse_extents[1]
            else:
                s[op].bind(fused_spatial_axes[0], tvm.thread_axis("blockIdx.x"))
                s[op].bind(fused_spatial_axes[1], tvm.thread_axis("vthread"))
                s[op].bind(fused_spatial_axes[2], tvm.thread_axis("threadIdx.x"))
                thread_pos = fused_spatial_axes[2]
                thread_extents = spatial_fuse_extents[2]

            # always compute at here
            s[write_cache].compute_at(s[op], thread_pos)

            # reduce_split
            reduced_axes = s[write_cache].op.reduce_axis
            splited_reduced_axes = []
            if "reduce" in config and len(config["reduce"]) > 0:
                # to align each axis
                assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
                for axis, nparts in zip(reduced_axes, config["reduce"]):
                    tmp_buffer = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_reduced_axes.append(tmp_buffer)
            else:
                for axis in reduced_axes:
                    splited_reduced_axes.append([axis])
            share_pos = None
            local_pos = None
            # if has reduce axes
            if len(splited_reduced_axes) > 0:
                # always reorder here
                reduced_nonfuse_lsts = []
                reorder_lst = []
                length = len(splited_reduced_axes[0])
                
                for count in range(length):
                    tmp_buffer = [x[count] for x in splited_reduced_axes]
                    reduced_nonfuse_lsts.append(tmp_buffer)
                    reorder_lst.extend(tmp_buffer)
                # change the order of reduce axes and spatial axes
                reorder_lst.extend(s[write_cache].op.axis)
                s[write_cache].reorder(*reorder_lst)

                if length == 1:
                    share_pos = reduced_nonfuse_lsts[-1][0]
                else:
                    share_pos = reduced_nonfuse_lsts[-2][0]
                    local_pos = reduced_nonfuse_lsts[-1][-1]

            # always cache read here
            if share_pos is not None:
                for share in read_cache_share_lst:
                    s[share].compute_at(s[write_cache], share_pos)
            else:
                for share in read_cache_share_lst:
                    s[share].compute_inline()
            if local_pos is not None:
                for local in read_cache_local_lst:
                    s[local].compute_at(s[write_cache], local_pos)
            else:
                for local in read_cache_local_lst:
                    s[local].compute_inline()
            
            # always cooperative fetching
            if share_pos is not None:
                for share in read_cache_share_lst:
                    fuse_lst = s[share].op.axis
                    fused = s[share].fuse(*fuse_lst)
                    outer, inner = s[share].split(fused, nparts=thread_extents)
                    s[share].bind(outer, tvm.thread_axis("threadIdx.x"))
            
            # unroll
            if "unroll" in config and len(config["unroll"]) > 0:
                step = config["unroll"][0][0]
                explicit = config["unroll"][0][1]
                s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
                s[op].pragma(kernel_scope, 'unroll_explicit', explicit)
        
        def _cuda_schedule_fuse_split(s, op, op_state):
            # assert_print(op in s)

            # always cache write here
            if op.num_outputs > 1:
                raise RuntimeWarning("Too many outputs in one operation!")
            write_cache = s.cache_write(op.output(0), "local")

            # always cache read here
            read_cache_share_lst = []
            # read_cache_local_lst = []
            for t in op.input_tensors:
                share = s.cache_read(t, "shared", [write_cache])
                read_cache_share_lst.append(share)
                # local = s.cache_read(share, "local", [write_cache])
                # read_cache_local_lst.append(local)
            
            # spatial fuse
            spatial_axes = s[op].op.axis
            fused_spatial_axes = []
            if "fuse" in config and len(config["fuse"]) > 0:
                # fuse redundant axes
                beg = 0
                for end in config["fuse"][0]:
                    fuse_lst = spatial_axes[beg:end]
                    beg = end
                    if len(fuse_lst) > 0:
                        fused = s[op].fuse(*fuse_lst)
                        fused_spatial_axes.append(fused)
            else:
                fused_spatial_axes = spatial_axes

            # spatial split
            split_factor_lst = []
            splited_spatial_axes = []
            if "spatial" in config and len(config["spatial"]) > 0:
                # to align each axis
                assert len(config["spatial"]) == len(spatial_axes), "align failed"
                # compute split factors
                if "fuse" in config and len(config["fuse"]) > 0:
                    beg = 0
                    for end in config["fuse"][0]:
                        tmp_lst = [1] * len(config["spatial"][0])
                        for i in range(beg, end):
                            for j in range(len(config["spatial"][i])):
                                tmp_lst[j] *= config["spatial"][i][j]
                        if beg < end:
                            split_factor_lst.append(tmp_lst)
                        beg = end
                else:
                    split_factor_lst = config["spatial"]
                assert len(fused_spatial_axes) == len(split_factor_lst), "align failed"
                for axis, nparts in zip(fused_spatial_axes, split_factor_lst):
                    tmp_buffer = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[op].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_spatial_axes.append(tmp_buffer)
            else:
                for axis in fused_spatial_axes:
                    splited_spatial_axes.append([axis])
            assert len(splited_spatial_axes) > 0, "empty spatial axes"     # must be non-empty

            # always reorder here
            reorder_lst = []
            for count in range(len(splited_spatial_axes[0])):
                tmp_buffer = [x[count] for x in splited_spatial_axes]
                reorder_lst.extend(tmp_buffer)
            s[op].reorder(*reorder_lst)

            # fix kernel scope
            kernel_scope = reorder_lst[0]
            
            # always bind here
            # - prepare thread axis
            bx = tvm.thread_axis("blockIdx.x")
            by = tvm.thread_axis("blockIdx.y")
            bz = tvm.thread_axis("blockIdx.z")
            vx = tvm.thread_axis("vthread")
            vy = tvm.thread_axis("vthread")
            vz = tvm.thread_axis("vthread")
            tx = tvm.thread_axis("threadIdx.x")
            ty = tvm.thread_axis("threadIdx.y")
            tz = tvm.thread_axis("threadIdx.z")

            blocks = [bz, by, bx]
            threads = [tz, ty, tx]
            vthreads = [vz, vy, vx]

            block_extents = [-1, -1, -1]    # z, y, x
            virtual_extents = [-1, -1, -1]
            thread_extents = [-1, -1, -1]

            length = len(splited_spatial_axes)
            assert length >= 1
            # - bind
            count = min(length, len(blocks)) - 1
            while count >= 0:
                parts = len(splited_spatial_axes[count])
                assert parts > 0
                if parts == 1:
                    s[op].bind(splited_spatial_axes[count][0], blocks[count])
                    block_extents[count] = split_factor_lst[count][0]
                elif parts == 2:
                    s[op].bind(splited_spatial_axes[count][0], blocks[count])
                    block_extents[count] = split_factor_lst[count][0]
                    s[op].bind(splited_spatial_axes[count][1], threads[count])
                    thread_extents[count] = split_factor_lst[count][1]
                else:
                    s[op].bind(splited_spatial_axes[count][0], blocks[count])
                    block_extents[count] = split_factor_lst[count][0]
                    s[op].bind(splited_spatial_axes[count][1], vthreads[count])
                    virtual_extents[count] = split_factor_lst[count][1]
                    s[op].bind(splited_spatial_axes[count][2], threads[count])
                    thread_extents[count] = split_factor_lst[count][2]
                count -= 1
            # - compute at pos
            count = min(length, len(blocks)) - 1
            parts = len(splited_spatial_axes[count])
            thread_pos = splited_spatial_axes[count][min(parts - 1, 2)]

            # always compute at here
            s[write_cache].compute_at(s[op], thread_pos)

            # reduce_split
            reduced_axes = s[write_cache].op.reduce_axis
            splited_reduced_axes = []
            if "reduce" in config and len(config["reduce"]) > 0:
                # to align each axis
                assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
                for axis, nparts in zip(reduced_axes, config["reduce"]):
                    tmp_buffer = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_reduced_axes.append(tmp_buffer)
            else:
                for axis in reduced_axes:
                    splited_reduced_axes.append([axis])
            share_pos = None
            # local_pos = None
            # if has reduce axes
            if len(splited_reduced_axes) > 0:
                # always reorder here
                reduced_nonfuse_lsts = []
                reorder_lst = []
                length = len(splited_reduced_axes[0])
                # leave the last part
                for count in range(length - 1):
                    tmp_buffer = [x[count] for x in splited_reduced_axes]
                    reduced_nonfuse_lsts.append(tmp_buffer)
                    reorder_lst.extend(tmp_buffer)
                # the last part
                last_part = [x[length - 1] for x in splited_reduced_axes]
                spatial_remainder = s[write_cache].op.axis
                # change the order of reduce axes and spatial axes
                if "reorder" in config and len(config["reorder"]) > 0:
                    pos = config["reorder"][0][0]
                    assert pos < len(spatial_remainder)
                    tmp_buffer = []
                    count = len(spatial_remainder) - 1
                    while count > pos:
                        tmp_buffer.append(spatial_remainder[count])
                        count -= 1
                    p = pos
                    q = len(last_part) - 1
                    while p >= 0 and q >= 0:
                        tmp_buffer.append(spatial_remainder[p])
                        tmp_buffer.append(last_part[q])
                        p -= 1
                        q -= 1
                    while p >= 0:
                        tmp_buffer.append(spatial_remainder[p])
                        p -= 1
                    while q >= 0:
                        tmp_buffer.append(last_part[q])
                        q -= 1
                    tmp_buffer = list(reversed(tmp_buffer))
                    reorder_lst.extend(tmp_buffer)
                else:
                    reorder_lst.extend(last_part)
                    reorder_lst.extend(spatial_remainder)
                s[write_cache].reorder(*reorder_lst)
                # decide where to compute at
                if length == 1:
                    share_pos = last_part[-1]
                else:
                    mid = math.ceil(length / 2.0) - 1
                    share_pos = reduced_nonfuse_lsts[mid][-1]
                    # local_pos = last_part[-1]

            # always cache read here
            if share_pos is not None:
                for share in read_cache_share_lst:
                    s[share].compute_at(s[write_cache], share_pos)
            else:
                for share in read_cache_share_lst:
                    s[share].compute_inline()
            # if local_pos is not None:
            #     for local in read_cache_local_lst:
            #         s[local].compute_at(s[write_cache], local_pos)
            # else:
            #     for local in read_cache_local_lst:
            #         s[local].compute_inline()
            
            # always cooperative fetching
            if share_pos is not None:
                for share in read_cache_share_lst:
                    fuse_lst = s[share].op.axis
                    fused = s[share].fuse(*fuse_lst)
                    count = 2
                    cur = 1
                    limit = 1024
                    while count >= 0:
                        factor = thread_extents[count]
                        if factor < 0:
                            defined = False
                            factor = 16
                        else:
                            defined = True
                        cur *= factor
                        if not defined and cur > limit:
                            break
                        fused, inner = s[share].split(fused, factor=factor)
                        s[share].bind(inner, threads[count])
                        count -= 1
            
            # unroll
            if "unroll" in config and len(config["unroll"]) > 0:
                step = config["unroll"][0][0]
                explicit = config["unroll"][0][1]
                s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
                s[op].pragma(kernel_scope, 'unroll_explicit', explicit)
        
        def _cuda_schedule_split_reorder_fuse(s, op, op_state):
            # assert_print(op in s)

            loop_lst = []
            loop_idx = []

            # always cache write here
            if op.num_outputs > 1:
                raise RuntimeWarning("Too many outputs in one operation!")
            write_cache = s.cache_write(op.output(0), "local")

            # always cache read here
            read_cache_share_lst = []
            # read_cache_local_lst = []
            for t in op.input_tensors:
                share = s.cache_read(t, "shared", [write_cache])
                read_cache_share_lst.append(share)
                # local = s.cache_read(share, "local", [write_cache])
                # read_cache_local_lst.append(local)

            # spatial split
            spatial_axes = [axis for axis in s[op].op.axis]
            assert len(spatial_axes) > 0, "empty spatial axes"     # must be non-empty
            n = spatial_axes[0]
            kernel_scope, n = s[op].split(n, nparts=1)
            spatial_axes[0] = n

            splited_spatial_axes = []
            splited_spatial_extents = []
            if "spatial" in config and len(config["spatial"]) > 0:
                # to align each axis
                assert len(config["spatial"]) == len(spatial_axes), "align failed"
                for axis, nparts in zip(spatial_axes, config["spatial"]):
                    tmp_buffer = []
                    tmp_extents = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[op].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                        tmp_extents.append(nparts[count])
                    tmp_buffer.append(axis)
                    tmp_extents.append(nparts[-1])
                    splited_spatial_axes.append(tmp_buffer)
                    splited_spatial_extents.append(tmp_extents)
            else:
                for axis in spatial_axes:
                    splited_spatial_axes.append([axis])
                    splited_spatial_extents.append([axis.dom.extent.value])

            # always reorder here
            reorder_lst = []
            reorder_parts = []
            reorder_part_extents = []
            for count in range(len(splited_spatial_axes[0])):
                tmp_buffer = [x[count] for x in splited_spatial_axes]
                tmp_extents = [x[count] for x in splited_spatial_extents]
                reorder_lst.extend(tmp_buffer)
                reorder_parts.append(tmp_buffer)
                reorder_part_extents.append(tmp_extents)
            s[op].reorder(*reorder_lst)

            # handle fuse request
            fused_parts = []
            fused_part_extents = []
            fused_part_idx = []
            if "fuse" in config and len(config["fuse"]) > 0:
                base_id = 0
                for part, extents in zip(reorder_parts, reorder_part_extents):
                    tmp_part = []
                    tmp_extents = []
                    tmp_idx = []
                    idx = 0
                    beg = 0
                    for end in config["fuse"][0]:
                        if end - beg > 1:
                            fuse_lst = part[beg:end]
                            fused = s[op].fuse(*fuse_lst)
                            tmp_part.append(fused)
                            extent = reduce(lambda x, y: x * y, extents[beg:end], 1)
                            tmp_idx.extend([idx] * (end - beg))
                            idx += 1
                            tmp_extents.append(extent)
                        elif end - beg == 1:
                            tmp_part.append(part[beg])
                            tmp_extents.append(extents[beg])
                            tmp_idx.append(idx)
                            idx += 1
                        beg = end
                    fused_parts.append(tmp_part)
                    fused_part_extents.append(tmp_extents)
                    fused_part_idx.append(tmp_idx)

                    loop_lst.extend(tmp_part)
                    loop_idx.extend([x + base_id for x in tmp_idx])
                    base_id += len(tmp_part)
            else:
                fused_parts = reorder_parts
                fused_part_extents = reorder_part_extents
                fused_part_idx = [list(range(len(x))) for x in reorder_parts]

                loop_lst = reorder_lst
                loop_idx = list(range(len(reorder_lst)))

            # record op state
            op_state.loop_lst = loop_lst
            op_state.loop_idx = loop_idx
      
            # always bind here
            # - prepare thread axis
            bx = tvm.thread_axis("blockIdx.x")
            by = tvm.thread_axis("blockIdx.y")
            bz = tvm.thread_axis("blockIdx.z")
            vx = tvm.thread_axis("vthread")
            vy = tvm.thread_axis("vthread")
            vz = tvm.thread_axis("vthread")
            tx = tvm.thread_axis("threadIdx.x")
            ty = tvm.thread_axis("threadIdx.y")
            tz = tvm.thread_axis("threadIdx.z")

            blocks = [bz, by, bx]
            threads = [tz, ty, tx]
            vthreads = [vz, vy, vx]

            block_extents = [-1, -1, -1]    # z, y, x
            virtual_extents = [-1, -1, -1]
            thread_extents = [-1, -1, -1]

            bind_option = [None, None, None]
            bind_candidate = [blocks, vthreads, threads]
            candiate_extents = [block_extents, virtual_extents, thread_extents]

            # - bind
            num_parts = len(fused_parts)
            if num_parts == 1:
                bind_option[0] = (fused_parts[0], fused_part_extents[0])
                local_pos = fused_parts[0][:len(bind_candidate[0])][-1]
            elif num_parts == 2:
                bind_option[0] = (fused_parts[0], fused_part_extents[0])
                bind_option[2] = (fused_parts[1], fused_part_extents[1])
                local_pos = fused_parts[1][:len(bind_candidate[2])][-1]
            else:
                bind_option[0] = (fused_parts[0], fused_part_extents[0])
                bind_option[1] = (fused_parts[1], fused_part_extents[1])
                bind_option[2] = (fused_parts[2], fused_part_extents[2])
                local_pos = fused_parts[2][:len(bind_candidate[2])][-1]
            for option, candidate, extents in zip(bind_option, bind_candidate, candiate_extents):
                if option is not None:
                    for i, axis in enumerate(option[0][:len(candidate)]):
                        s[op].bind(axis, candidate[i])
                        extents[i] = option[1][i]
                    
            # compute at
            if "local_pos" in config and len(config["local_pos"]) > 0:
                local_at_part = config["local_pos"][0][0]
                local_at_idx = config["local_pos"][0][1]
                # index changed because of fusion
                cur_idx = fused_part_idx[local_at_part][local_at_idx]
                local_pos = fused_parts[local_at_part][cur_idx]

            # always compute at here
            s[write_cache].compute_at(s[op], local_pos)

            # reduce_split
            reduced_axes = s[write_cache].op.reduce_axis
            splited_reduced_axes = []
            if "reduce" in config and len(config["reduce"]) > 0:
                # to align each axis
                assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
                for axis, nparts in zip(reduced_axes, config["reduce"]):
                    tmp_buffer = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_reduced_axes.append(tmp_buffer)
            else:
                for axis in reduced_axes:
                    splited_reduced_axes.append([axis])
            share_pos = None
            # local_pos = None
            # if has reduce axes
            if len(splited_reduced_axes) > 0:
                # always reorder here
                reduced_nonfuse_lsts = []
                reorder_lst = []
                length = len(splited_reduced_axes[0])
                # leave the last part
                for count in range(length - 1):
                    tmp_buffer = [x[count] for x in splited_reduced_axes]
                    reduced_nonfuse_lsts.append(tmp_buffer)
                    reorder_lst.extend(tmp_buffer)
                # the last part
                last_part = [x[length - 1] for x in splited_reduced_axes]
                spatial_remainder = s[write_cache].op.axis
                # change the order of reduce axes and spatial axes
                if "reorder" in config and len(config["reorder"]) > 0:
                    pos = config["reorder"][0][0]
                    assert pos < len(spatial_remainder)
                    tmp_buffer = []
                    count = len(spatial_remainder) - 1
                    while count > pos:
                        tmp_buffer.append(spatial_remainder[count])
                        count -= 1
                    p = pos
                    q = len(last_part) - 1
                    while p >= 0 and q >= 0:
                        tmp_buffer.append(spatial_remainder[p])
                        tmp_buffer.append(last_part[q])
                        p -= 1
                        q -= 1
                    while p >= 0:
                        tmp_buffer.append(spatial_remainder[p])
                        p -= 1
                    while q >= 0:
                        tmp_buffer.append(last_part[q])
                        q -= 1
                    tmp_buffer = list(reversed(tmp_buffer))
                    reorder_lst.extend(tmp_buffer)
                else:
                    reorder_lst.extend(last_part)
                    reorder_lst.extend(spatial_remainder)
                s[write_cache].reorder(*reorder_lst)
                # decide where to compute at
                if "share_pos" in config and len(config["share_pos"]) > 0:
                    share_at = config["share_pos"][0][0]
                    share_idx = config["share_pos"][0][1]
                    reduced_nonfuse_lsts.append(last_part)
                    share_pos = reduced_nonfuse_lsts[share_at][share_idx]
                else:
                    if length == 1:
                        share_pos = last_part[-1]
                    else:
                        mid = math.ceil(length / 2.0) - 1
                        share_pos = reduced_nonfuse_lsts[mid][-1]
                        # local_pos = last_part[-1]

            # always cache read here
            if share_pos is not None:
                for share in read_cache_share_lst:
                    s[share].compute_at(s[write_cache], share_pos)
            else:
                for share in read_cache_share_lst:
                    s[share].compute_inline()
            # if local_pos is not None:
            #     for local in read_cache_local_lst:
            #         s[local].compute_at(s[write_cache], local_pos)
            # else:
            #     for local in read_cache_local_lst:
            #         s[local].compute_inline()
            
            # always cooperative fetching
            if share_pos is not None:
                for share in read_cache_share_lst:
                    fuse_lst = s[share].op.axis
                    fused = s[share].fuse(*fuse_lst)
                    count = 2
                    cur = 1
                    limit = 1024
                    while count >= 0:
                        factor = thread_extents[count]
                        if factor < 0:
                            defined = False
                            factor = 16
                        else:
                            defined = True
                        cur *= factor
                        if not defined and cur > limit:
                            break
                        fused, inner = s[share].split(fused, factor=factor)
                        s[share].bind(inner, threads[count])
                        count -= 1
            
            # unroll
            if "unroll" in config and len(config["unroll"]) > 0:
                step = config["unroll"][0][0]
                explicit = config["unroll"][0][1]
                s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
                s[op].pragma(kernel_scope, 'unroll_explicit', explicit)

        def _cpu_schedule_split_fuse(s, op, op_state):
            # always cache write here
            if op.num_outputs > 1:
                raise RuntimeWarning("Too many outputs in one operation!")
            write_cache = s.cache_write(op.output(0), "global")

            # spatial split
            spatial_axes = s[op].op.axis
            splited_spatial_axes = []
            if "spatial" in config and len(config["spatial"]) > 0:
                # to align each axis
                assert_print(len(config["spatial"]) == len(spatial_axes), "align failed")
                for axis, nparts in zip(spatial_axes, config["spatial"]):
                    tmp_buffer = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[op].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_spatial_axes.append(tmp_buffer)
            else:
                for axis in spatial_axes:
                    splited_spatial_axes.append([axis])
            assert_print(len(splited_spatial_axes) > 0, "empty spatial axes")     # must be non-empty

            # always reorder and fuse here
            spatial_fuse_lsts = []
            spatial_fuse_extents = []
            reorder_lst = []
            fused_spatial_axes = []
            for count in range(len(splited_spatial_axes[0])):
                tmp_buffer = [x[count] for x in splited_spatial_axes]
                tmp_extent = reduce(lambda a, b: a * b, [x[count] for x in config["spatial"]])
                spatial_fuse_lsts.append(tmp_buffer)
                spatial_fuse_extents.append(tmp_extent)
                reorder_lst.extend(tmp_buffer)
            s[op].reorder(*reorder_lst)
            for fuse_lst in spatial_fuse_lsts:
                fused = s[op].fuse(*fuse_lst)
                fused_spatial_axes.append(fused)
            kernel_scope = fused_spatial_axes[0]
            
            # always parallel here
            length = len(fused_spatial_axes)
            assert_print(length > 0, "empty spatial axes!")
            s[op].parallel(fused_spatial_axes[0])
            if length == 1:
                thread_pos = fused_spatial_axes[0]
            if 2 <= length <= 3:
                thread_pos = fused_spatial_axes[1]
            else:
                thread_pos = fused_spatial_axes[2]

            # always compute at here
            s[write_cache].compute_at(s[op], thread_pos)

            # reduce_split
            reduced_axes = s[write_cache].op.reduce_axis
            splited_reduced_axes = []
            if "reduce" in config and len(config["reduce"]) > 0:
                # to align each axis
                assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
                for axis, nparts in zip(reduced_axes, config["reduce"]):
                    tmp_buffer = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_reduced_axes.append(tmp_buffer)
            else:
                for axis in reduced_axes:
                    splited_reduced_axes.append([axis])

            # if has reduce axes
            if len(splited_reduced_axes) > 0:
                # always reorder here
                reduced_nonfuse_lsts = []
                reorder_lst = []
                length = len(splited_reduced_axes[0])
                
                for count in range(length):
                    tmp_buffer = [x[count] for x in splited_reduced_axes]
                    reduced_nonfuse_lsts.append(tmp_buffer)
                    reorder_lst.extend(tmp_buffer)
                # change the order of reduce axes and spatial axes
                rlength = len(splited_reduced_axes)
                if rlength > 1:
                    reorder_lst.extend(s[write_cache].op.axis)
                elif rlength == 1:   # in this case, have to interleave otherwise the reorder is of no use
                    tmp_order = []
                    p_spatial = len(s[write_cache].op.axis) - 1
                    p_reduce = len(reorder_lst) - 1
                    while p_spatial >= 0 and p_reduce >= 0:
                        tmp_order.append(s[write_cache].op.axis[p_spatial])
                        tmp_order.append(reorder_lst[p_reduce])
                        p_spatial -= 1
                        p_reduce -= 1
                    while p_spatial >= 0:
                        tmp_order.append(s[write_cache].op.axis[p_spatial])
                        p_spatial -= 1
                    while p_reduce >= 0:
                        tmp_order.append(reorder_lst[p_reduce])
                        p_reduce -= 1
                    tmp_order = list(reversed(tmp_order))
                    reorder_lst = tmp_order
                s[write_cache].reorder(*reorder_lst)
            
            # unroll
            if "unroll" in config and len(config["unroll"]) > 0:
                step = config["unroll"][0][0]
                s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
            
            # always vectorize here
            # s[write_cache].vectorize(s[write_cache].op.axis[-1])
        
        def _cpu_schedule_split_reorder_fuse(s, op, op_state):
            # assert_print(op in s)

            loop_idx = []
            loop_lst = []

            # always cache write here
            if op.num_outputs > 1:
                raise RuntimeWarning("Too many outputs in one operation!")
            write_cache = s.cache_write(op.output(0), "local")

            # spatial split
            spatial_axes = [axis for axis in s[op].op.axis]
            assert len(spatial_axes) > 0, "empty spatial axes"     # must be non-empty
            n = spatial_axes[0]
            kernel_scope, n = s[op].split(n, nparts=1)
            spatial_axes[0] = n

            splited_spatial_axes = []
            splited_spatial_extents = []
            if "spatial" in config and len(config["spatial"]) > 0:
                # to align each axis
                assert len(config["spatial"]) == len(spatial_axes), "align failed"
                for axis, nparts in zip(spatial_axes, config["spatial"]):
                    tmp_buffer = []
                    tmp_extents = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[op].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                        tmp_extents.append(nparts[count])
                    tmp_buffer.append(axis)
                    tmp_extents.append(nparts[-1])
                    splited_spatial_axes.append(tmp_buffer)
                    splited_spatial_extents.append(tmp_extents)
            else:
                for axis in spatial_axes:
                    splited_spatial_axes.append([axis])
                    splited_spatial_extents.append([axis.dom.extent.value])

            # always reorder here
            reorder_lst = []
            reorder_parts = []
            reorder_part_extents = []
            for count in range(len(splited_spatial_axes[0])):
                tmp_buffer = [x[count] for x in splited_spatial_axes]
                tmp_extents = [x[count] for x in splited_spatial_extents]
                reorder_lst.extend(tmp_buffer)
                reorder_parts.append(tmp_buffer)
                reorder_part_extents.append(tmp_extents)
            s[op].reorder(*reorder_lst)

            # handle fuse request
            fused_parts = []
            fused_part_extents = []
            fused_part_idx = []
            if "fuse" in config and len(config["fuse"]) > 0:
                base_id = 0
                for part, extents in zip(reorder_parts, reorder_part_extents):
                    tmp_part = []
                    tmp_extents = []
                    tmp_idx = []
                    idx = 0
                    beg = 0
                    for end in config["fuse"][0]:
                        if end - beg > 1:
                            fuse_lst = part[beg:end]
                            fused = s[op].fuse(*fuse_lst)
                            tmp_part.append(fused)
                            extent = reduce(lambda x, y: x * y, extents[beg:end], 1)
                            tmp_idx.extend([idx] * (end - beg))
                            idx += 1
                            tmp_extents.append(extent)
                        elif end - beg == 1:
                            tmp_part.append(part[beg])
                            tmp_extents.append(extents[beg])
                            tmp_idx.append(idx)
                            idx += 1
                        beg = end
                    fused_parts.append(tmp_part)
                    fused_part_extents.append(tmp_extents)
                    fused_part_idx.append(tmp_idx)

                    # for op state
                    loop_lst.extend(tmp_part)
                    loop_idx.extend([x + base_id for x in tmp_idx])
                    base_id += len(tmp_part)
            else:
                fused_parts = reorder_parts
                fused_part_extents = reorder_part_extents
                fused_part_idx = [list(range(len(x))) for x in reorder_parts]

                # for op state
                loop_lst = reorder_lst
                loop_idx = list(range(len(reorder_lst)))

            # record op state
            op_state.loop_lst = loop_lst
            op_state.loop_idx = loop_idx

            # parallel
            fused = s[op].fuse(*fused_parts[0])
            s[op].parallel(fused)
     
            # compute at
            num_parts = len(fused_parts)
            if num_parts == 1:
                local_pos = fused
            elif num_parts == 2:
                local_pos = fused_parts[num_parts-1][0]
            else:
                local_pos = fused_parts[num_parts-2][-1]

            if "local_pos" in config and len(config["local_pos"]) > 0:
                local_at_part = config["local_pos"][0][0]
                local_at_idx = config["local_pos"][0][1]
                # index changed because of fusion
                cur_idx = fused_part_idx[local_at_part][local_at_idx]
                local_pos = fused_parts[local_at_part][cur_idx]

            # always compute at here
            s[write_cache].compute_at(s[op], local_pos)

            # reduce_split
            reduced_axes = s[write_cache].op.reduce_axis
            splited_reduced_axes = []
            if "reduce" in config and len(config["reduce"]) > 0:
                # to align each axis
                assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
                for axis, nparts in zip(reduced_axes, config["reduce"]):
                    tmp_buffer = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_reduced_axes.append(tmp_buffer)
            else:
                for axis in reduced_axes:
                    splited_reduced_axes.append([axis])

            # if has reduce axes
            if len(splited_reduced_axes) > 0:
                # always reorder here
                reduced_nonfuse_lsts = []
                reorder_lst = []
                length = len(splited_reduced_axes[0])
                # leave the last part
                for count in range(length - 1):
                    tmp_buffer = [x[count] for x in splited_reduced_axes]
                    reduced_nonfuse_lsts.append(tmp_buffer)
                    reorder_lst.extend(tmp_buffer)
                # the last part
                last_part = [x[length - 1] for x in splited_reduced_axes]
                spatial_remainder = s[write_cache].op.axis
                # change the order of reduce axes and spatial axes
                if "reorder" in config and len(config["reorder"]) > 0:
                    pos = config["reorder"][0][0]
                    assert pos < len(spatial_remainder)
                    tmp_buffer = []
                    count = len(spatial_remainder) - 1
                    while count > pos:
                        tmp_buffer.append(spatial_remainder[count])
                        count -= 1
                    p = pos
                    q = len(last_part) - 1
                    while p >= 0 and q >= 0:
                        tmp_buffer.append(spatial_remainder[p])
                        tmp_buffer.append(last_part[q])
                        p -= 1
                        q -= 1
                    while p >= 0:
                        tmp_buffer.append(spatial_remainder[p])
                        p -= 1
                    while q >= 0:
                        tmp_buffer.append(last_part[q])
                        q -= 1
                    tmp_buffer = list(reversed(tmp_buffer))
                    reorder_lst.extend(tmp_buffer)
                else:
                    reorder_lst.extend(last_part)
                    reorder_lst.extend(spatial_remainder)
                s[write_cache].reorder(*reorder_lst)
            
            # unroll
            if "unroll" in config and len(config["unroll"]) > 0:
                step = config["unroll"][0][0]
                explicit = config["unroll"][0][1]
                s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
                s[op].pragma(kernel_scope, 'unroll_explicit', explicit)

        if target == "cuda":
            # if hint == "split_fuse":
            #     print(hint)
            #     return _cuda_schedule_split_fuse
            # elif hint == "fuse_split":
            #     print(hint)
            #     return _cuda_schedule_fuse_split
            # else:
            #     raise RuntimeError("Unknown hint: %s" % hint)
            return _cuda_schedule_split_reorder_fuse
        elif target == "llvm":
            return _cpu_schedule_split_reorder_fuse
        else:
            raise RuntimeError("Currently no support for target %s"%target)  


class GraphScheduler(Scheduler):
    def __init__(self, task_key, space, decay=0.7, parallel=10, timeout=4.0, trial=100, number=10, early_stop=30, rpc_info=None):
        super(GraphScheduler, self).__init__("graph", task_key, space, parallel, timeout, trial, number, early_stop, rpc_info)

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
                                continue    # do not compute at inlined ops
                            consumer_loop_idx = consumer_state.loop_idx
                            at_pos = consumer_state.loop_lst[consumer_loop_idx[entity[count]]]
                            s[op_lst[count]].compute_at(s[op_lst[consumer_id]], at_pos)
                            op_states[count].compute_at = True

        if phase == "inline":
            return _inline_schedule
        elif phase == "at":
            return _at_schedule
        else:
            raise RuntimeError("Currently no support for phase %s" %phase)
        


class Result(object):
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def get(self, timeout=1):
        # beg = time.time()
        # while time.time() - beg < timeout:
        #     if self.q.empty():
        #         time.sleep(.1)
        #     else:
        #         break
        try:
            # print("getting...")
            # while self.q.empty():
            #     pass
            # print("queue is empty? ", self.q.empty())
            res = self.q.get(block=True, timeout=timeout)
            # print("done")
            # while not self.q.empty():
            #     _ = self.q.get(block=True)
        except Exception as e:
            # print(e.__class__)
            res = RuntimeError(str(e))
        if self.p.is_alive():
            kill_child_processes(self.p.pid)
            self.p.terminate()
        self.p.join()
        self.q.close()
        # print("queue joining...")
        self.q.join_thread()
        # print("queue joined")
        del self.p
        del self.q
        return res


class OpState(object):
    def __init__(self):
        self.inline = False
        self.loop_lst = []
        self.loop_idx = []
        self.compute_at = False
        self.consumer_lst = []


class RpcInfo(object):
    def __init__(self, host, port, target_host=None):
        self.host = host
        self.port = port
        self.target_host = target_host


def schedule(task_key, slevel=4, rlevel=3, op_trial=50, graph_trial=10, op_stop=15, graph_stop=5, 
        number=10, timeout=5.0, parallel=8, method="searching", **kwargs):
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
    rpc_info = None
    if "rpc_info" in kwargs:
        rpc_info = kwargs["rpc_info"]
    ##################################################
    # first generate graph space
    graph_space = generate_space_inter_op(op_lst, down_graph, force_inline=force_inline)

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
        if task.target == "cuda":
            space = generate_space_intra_op(op, down_graph, slevel=slevel, groups=3)
        elif task.target == "llvm":
            space = generate_space_intra_op(op, down_graph, slevel=slevel, rlevel=rlevel)
        else:
            raise RuntimeError("Currently no support for target %s"%task.target)
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
            rpc_info=rpc_info
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
        rpc_info=rpc_info
        )
    use_model = False if graph_perf_model_path is None else True
    graph_config = graph_scheduler.schedule(configs, method=method, use_model=use_model, perf_path=graph_perf_model_path)
    #################################################
    # combine the configs
    configs = Config(configs.op_config_lst, graph_config)
    
    #################################################
    # final schedule
    s = tvm.create_schedule(ops)
    # perform inter operator schedule
    graph_template = GraphScheduler.generate_graph_schedule(configs.graph_config, phase="inline")
    graph_template(s, op_lst, op_states)
    # perform intra-operator schedule
    for count_op, (op, op_state, op_config) in enumerate(zip(op_lst, op_states, configs.op_config_lst)):
        if not op_state.inline:
            op_template = OpScheduler.generate_op_schedule(task.target, op_config)
            op_template(s, op, op_states[count_op])
    # perform inter operations schedule again for compute at
    if graph_config is not None:
        graph_template = GraphScheduler.generate_graph_schedule(graph_config, phase="at")
        graph_template(s, op_lst, op_states)

    return s, bufs, configs
    

def schedule_with_config(task_key, configs, op_pos=None):
    """Schedule a task with given configs

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

    op_config_lst = configs.op_config_lst

    if op_pos is not None:
        assert_print(isinstance(op_pos, int), "op_pos should be int")
        assert_print(op_pos < len(op_lst) and op_pos < len(op_config_lst), "op_pos too big")
        loop_length = op_pos + 1
        s = tvm.create_schedule(op_lst[op_pos])
    else:
        assert_print(len(op_config_lst) <= len(op_lst), "config length exceed op_lst")
        loop_length = len(op_config_lst)
        s = tvm.create_schedule(ops)

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
            template = OpScheduler.generate_op_schedule(task.target, config)
            template(s, op, op_states[i])   

    ###################################################
    # perform inter operations schedule again for compute at
    if graph_config is not None:
        graph_template = GraphScheduler.generate_graph_schedule(graph_config, phase="at")
        graph_template(s, op_lst, op_states)

    return s, bufs

