import os
import time
import signal
import tvm
import numpy as np
import torch.multiprocessing as multi
from collections import deque, namedtuple
from functools import reduce
from auto_schedule.testing.task import TASK_TABLE
from auto_schedule.testing.model import Walker
from auto_schedule.testing.space import generate_space_inter_op, generate_space_intra_op
from auto_schedule.utils import assert_print, to_tuple
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
        if isinstance(cur, tvm.tensor.ComputeOp):
            bfs_order.append(cur)
        for t in cur.input_tensors:
            if t.op not in visited:
                visited.add(t.op)
                q.append(t.op)
            if t.op not in down_graph:
                down_graph[t.op] = []
            down_graph[t].append(cur)
    return list(reversed(bfs_order)), down_graph


def build_func(func_name, task_key, configs, op_pos=None):
    task = TASK_TABLE[task_key]
    s, bufs = schedule_with_config(task_key, configs, op_pos=op_pos)
    func = tvm.build(s, bufs, task.target)
    func.export_library(func_name)
    result = ([to_tuple(x.shape) for x in bufs], bufs[0].dtype)
    return result


def eval_func(func_file, bufs_shape, dtype, target, number=100, dev_id=0):
    ctx = tvm.context(target, dev_id)
    tvm_arys = []
    for shape in bufs_shape:
        shape = to_tuple(shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    func = tvm.module.load(func_file)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
    time_cost = evaluator(*tvm_arys).mean * 1e3
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
        res = func(*args, **kwargs)
    except Exception as e:
        res = e
    queue.put(res)


def parallel_execute(func, timeout, *args, **kwargs):
    q = multi.Queue(2)
    p = multi.Process(
        target=call_with_timeout, 
        args=(func, q, timeout, args, kwargs))
    p.start()
    return Result(p, q)


def call_with_timeout(func, queue, timeout, args, kwargs):
    p = multi.Process(target=exec_func, args=(func, queue, args, kwargs))
    p.start()
    p.join(timeout=timeout)

    queue.put(multi.TimeoutError())

    p.terminate()
    p.join()    


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


class OpScheduler(object):
    def __init__(self, task_key, op_pos, space, decay=0.7, parallel=10, timeout=4.0, trial=100, number=100):
        self.task_key = task_key
        self.op_pos = op_pos
        self.space = space
        self.parallel = parallel
        self.timeout = timeout
        self.trial = trial
        self.number = number

    def schedule(self, configs):
        walkers = {}
        accepted_proposals = {}
        behaviour = {}
        scope = self.parallel
        for (name, subspace) in self.space.items():
            walker = Walker(subspace, scope=scope)
            walkers[name] = walker
            behaviour[name] = 0.0
        # initialization
        init_proposals = {}
        pointers = {}
        for name, walker in walkers.items():
            proposal = walker.propose()
            assert_print(len(proposal) > 0, "Found initial proprosal empty!")
            init_proposals[name] = proposal
            pointers[name] = 0
        next_op_configs = []
        for i in range(scope):
            config = {}
            for type_key in self.space.valid_type_keys:
                config[type_key] = []
                for name in self.space.types[type_key]:
                    index = init_proposals[name][pointers[name]][0]
                    entity = self.space.subspaces[name].get_entity(index)
                    config[type_key].append(entity)
                    pointers[name] = (pointers[name] + 1) % len(init_proposals[name])
            # hack here
            if self.op_pos == 1:
                config = {
                    "spatial": [[1, 1, 1, 1], [64, 2, 8, 1], [1, 1, 7, 1], [1, 1, 7, 1]],
                    "reduce": [[64, 1, 16], [1, 3, 1], [1, 1, 3]],
                    "unroll": [[1024, 1]]
                }
            next_op_configs.append(config)
        initial_results = self.parallel_evaluate(configs, next_op_configs)
        # for testing, just choose one
        print(initial_results)
        choice = np.argmin(initial_results)
        return next_op_configs[choice]

    def parallel_evaluate(self, configs, next_op_configs):
        target = TASK_TABLE[self.task_key].target
        build_res_lst = []
        func_name = "tmp_{}.so".format(time.time())
        for config in next_op_configs:
            res = parallel_execute(
                build_func, 
                self.timeout, 
                func_name,
                self.task_key, 
                Config(configs.op_config_lst + [config], configs.graph_config), 
                self.op_pos)
            build_res_lst.append(res)
        
        eval_res_lst = []
        for build_res in build_res_lst:
            final_res = build_res.get()
            if isinstance(final_res, Exception):
                print("op build fail")
                eval_res_lst.append(float("inf"))
            else:
                res = parallel_execute(
                    eval_func,
                    self.timeout,
                    func_name,
                    final_res[0],
                    final_res[1],
                    target,
                    number=self.number,
                    dev_id=find_idle_device(target)
                )
                eval_res_lst.append(res)

        ret_lst = []
        for eval_res in eval_res_lst:
            if isinstance(eval_res, float):
                ret_lst.append(eval_res)
            else:
                final_res = eval_res.get()
                if isinstance(final_res, Exception):
                    # print("op run fail")
                    print(final_res)
                    ret_lst.append(float("inf"))
                else:
                    ret_lst.append(final_res)

        try:
            os.remove(func_name)
        except FileNotFoundError:
            pass

        return ret_lst


class GraphScheduler(object):
    def __init__(self, task_key, space, decay=0.7, parallel=10, timeout=4.0, trial=100, number=100):
        self.task_key = task_key
        self.space = space
        self.parallel = parallel
        self.timeout = timeout
        self.trial = trial
        self.number = number

    def schedule(self, configs):
        walkers = {}
        accepted_proposals = {}
        behaviour = {}
        scope = self.parallel
        for (name, subspace) in self.space.items():
            walker = Walker(subspace, scope=scope)
            walkers[name] = walker
            behaviour[name] = 0.0
        # initialization
        init_proposals = {}
        pointers = {}
        for name, walker in walkers.items():
            proposal = walker.propose()
            assert_print(len(proposal) > 0, "Found initial proprosal empty!")
            init_proposals[name] = proposal
            pointers[name] = 0
        graph_configs = []
        for i in range(scope):
            config = {}
            for type_key in self.space.valid_type_keys:
                config[type_key] = []
                for name in self.space.types[type_key]:
                    index = init_proposals[name][pointers[name]][0]
                    entity = self.space.subspaces[name].get_entity(index)
                    config[type_key].append(entity)
                    pointers[name] = (pointers[name] + 1) % len(init_proposals[name])
            # hack here
            config = {
                "inline": [[0]]
            }
            graph_configs.append(config)
        initial_results = self.parallel_evaluate(configs, graph_configs)
        # for testing, just choose one
        print(initial_results)
        choice = np.argmin(initial_results)
        return graph_configs[choice]

    def parallel_evaluate(self, configs, graph_configs):
        target = TASK_TABLE[self.task_key].target
        func_name = "graph_{}.so".format(time.time())
        build_res_lst = []
        for config in graph_configs:
            res = parallel_execute(
                build_func, 
                self.timeout, 
                func_name,
                self.task_key, 
                Config(configs.op_config_lst, config)
                )
            build_res_lst.append(res)
        
        eval_res_lst = []
        for build_res in build_res_lst:
            final_res = build_res.get()
            if isinstance(final_res, Exception):
                print("graph build fail")
                eval_res_lst.append(float("inf"))
            else:
                res = parallel_execute(
                    eval_func,
                    self.timeout,
                    func_name,
                    final_res[0],
                    final_res[1],
                    target,
                    number=self.number,
                    dev_id=find_idle_device(target)
                )
                eval_res_lst.append(res)
        
        ret_lst = []
        for eval_res in eval_res_lst:
            if isinstance(eval_res, float):
                ret_lst.append(eval_res)
            else:
                final_res = eval_res.get()
                if isinstance(final_res, Exception):
                    # print("graph run fail")
                    print(final_res)
                    ret_lst.append(float("inf"))
                else:
                    ret_lst.append(final_res)
        
        try:
            os.remove(func_name)
        except FileNotFoundError:
            pass

        return ret_lst
        


class Result(object):
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def get(self, timeout=None):
        try:
            res = self.q.get(block=True, timeout=timeout)
        except Exception as e:
            res = e
        if self.p.is_alive():
            kill_child_processes(self.p.pid)
            self.p.terminate()
        self.p.join()
        self.q.close()
        self.q.join_thread()
        del self.p
        del self.q
        return res


def generate_op_template(target, config):
    def _cuda_template(s, op):
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
        if "spatial" in config:
            # to align each axis
            assert_print(len(config["spatial"]) == len(spatial_axes))
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
        assert_print(len(splited_spatial_axes) > 0)     # must be non-empty

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
        assert_print(length > 1)
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
        if "reduce" in config:
            # to align each axis
            assert_print(len(config["reduce"]) == len(reduced_axes))
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
        if "unroll" in config:
            step = config["unroll"][0][0]
            explicit = config["unroll"][0][1]
            s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
            s[op].pragma(kernel_scope, 'unroll_explicit', explicit)

    if target == "cuda":
        return _cuda_template
    else:
        raise RuntimeError("Currently no support for target %s"%target)          


def generate_graph_template(config):
    def _common_template(s, op_lst, op_states):
        if "inline" in config:
            entity = config["inline"][0]
            for index in entity:
                if index >= 0:
                    s[op_lst[index]].compute_inline()
                    op_states[index].inline = True
    
    return _common_template


class OpState(object):
    def __init__(self):
        self.inline = False


def schedule(task_key):
    """Schedule a task

    perform sequential schedule
    """
    task = TASK_TABLE[task_key]
    func = task.func
    args = task.args
    ops, bufs = func(*args)
    # sort the ops, so that we can distinguish each op
    op_lst, down_graph = flatten_graph(ops)
    ##################################################
    # intra operations schedule decisionss
    op_space_lst = []
    configs = Config([], None)
    for pos, op in enumerate(op_lst):
        if task.target == "cuda":
            space = generate_space_intra_op(op, down_graph, level=4)
        else:
            raise RuntimeError("Currently no support for target %s"%task.target)
        op_space_lst.append(space)
        op_scheduler = OpScheduler(task_key, pos, space, parallel=1)
        op_config = op_scheduler.schedule(configs)
        configs.op_config_lst.append(op_config)

    #################################################
    # inter operations schedule decisions 
    graph_space = generate_space_inter_op(op_lst, down_graph)
    graph_scheduler = GraphScheduler(task_key, graph_space, parallel=1)
    graph_config = graph_scheduler.schedule(configs)

    #################################################
    # combine the configs
    configs = Config(configs.op_config_lst, graph_config)
    
    #################################################
    # final schedule
    s = tvm.create_schedule(ops)
    op_states = [OpState() for op in op_lst]
    # perform inter operator schedule
    graph_template = generate_graph_template(configs.graph_config)
    graph_template(s, op_lst, op_states)
    # perform intra-operator schedule
    for op, op_state, op_config in zip(op_lst, op_states, configs.op_config_lst):
        if not op_state.inline:
            op_template = generate_op_template(task.target, op_config)
            op_template(s, op)

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
    op_states = [OpState() for op in op_lst]

    op_config_lst = configs.op_config_lst

    if op_pos is not None:
        assert_print(isinstance(op_pos, int))
        assert_print(op_pos < len(op_lst) and op_pos < len(op_config_lst))
        loop_length = op_pos + 1
        s = tvm.create_schedule(op_lst[op_pos])
    else:
        assert_print(len(op_config_lst) <= len(op_lst))
        loop_length = len(op_config_lst)
        s = tvm.create_schedule(ops)

    ###################################################
    # perform inter operations schedule first
    graph_config = configs.graph_config
    if graph_config is not None:
        graph_template = generate_graph_template(graph_config)
        graph_template(s, op_lst, op_states)

    ###################################################
    # perform intra operations schedule    
    for i in range(loop_length):
        # mask inlined ops
        if not op_states[i].inline:
            op = op_lst[i]
            config = op_config_lst[i]
            template = generate_op_template(task.target, config)
            template(s, op)   

    return s, bufs

