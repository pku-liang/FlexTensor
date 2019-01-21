import torch
import numpy as np
import torch.optim as optim
import auto_schedule.Environment as Environment
from auto_schedule.config import *
from auto_schedule.training_examples import FUNC_TABLE
from auto_schedule import Env
from auto_schedule import ReverseBFSVisitor, SimpleEvaluator, NaturalScheduler


class TragetMessage(object):
    def __init__(self, target, l1_cache, l2_cache, l3_cache, block, thread):
        if target == "llvm":
            self.msg = [0, 0, l1_cache, l2_cache, l3_cache, block, thread]
        elif target == "cuda":
            self.msg = [1, 1, l1_cache, l2_cache, l3_cache, block, thread]
        if len(self.msg) < TARGET_MSG_LEN:
            res = TARGET_MSG_LEN - len(self.msg)
            for i in range(res):
                self.msg.append(0)
        self.msg = torch.FloatTensor(self.msg)


CPU_MSG = TragetMessage("llvm", 0.64, 2.56, 358.4, 1, 1)


def train_single_operator(compute, args, scheduler, target, target_msg):
    ops, arg_bufs = compute(*args)
    env = Env(ops)
    visitor = ReverseBFSVisitor()
    track = visitor.get_visit_msg(env).track
    evaluator = SimpleEvaluator()
    opt = optim.SGD(scheduler.parameters(), lr=0.02, momentum=0.5)

    loss_lst = []
    perform_lst = []
    mem = []
    for i in range(10):
        for j in range(16):
            scheduler.train()
            res = scheduler(track, env, ops, arg_bufs, target, target_msg)
            mem.append(res)
            if len(mem) > MEM_CPACITY:
                mem = mem[1:]
            sample_index = np.random.randint(0, 64 * len(mem), (4,))
            loss = 0.0
            for index in sample_index:
                outer = index // 64
                inner = index % 64
                sample = mem[outer]
                sm, op2schedule = sample.sm_lst[inner], sample.op2schedule_lst[inner]
                op2choice_cpat, op2choice_split = sample.op2choice_cpat_lst[inner], sample.op2choice_split_lst[inner]
                op2output_cpat, op2output_split = sample.op2output_cpat, sample.op2output_split
                sm.reset()
                for op in track:
                    node = env.op2node[op]
                    if not isinstance(node, Environment.ComputeVertex):
                        continue
                    before = evaluator.evaluate(sm.s, sm.arg_bufs, sm.target, dev_id=1)
                    sm.schedule_for(op, op2schedule[op])
                    after = evaluator.evaluate(sm.s, sm.arg_bufs, sm.target, dev_id=1)
                    reward = (before - after) / before
                    for choice in op2choice_cpat[op]:
                        loss = loss + (reward - op2output_cpat[op][0][choice]) ** 2
                    for k, choice in enumerate(op2choice_split[op]):
                        loss = loss + (reward - op2output_split[op][k][0][choice[0]]) ** 2
                        loss = loss + (reward - op2output_split[op][k][1][choice[1]]) ** 2
            opt.zero_grad()
            loss_lst.append(float(loss))
            loss.backward(retain_graph=True)
            opt.step()

            scheduler.eval()
            res = scheduler(track, env, ops, arg_bufs, target, target_msg)
            sm = res.sm_lst[0]
            performance = evaluator.evaluate(sm.s, sm.arg_bufs, sm.target, dev_id=1)
            perform_lst.append(performance)
            print("i=", i, "  j=", j, "  loss=", loss, "  performance=", performance)
    np.savetxt("loss.txt", np.array(loss_lst))
    np.savetxt("performance.txt", np.array(perform_lst))


if __name__ == "__main__":
    matmul, args = FUNC_TABLE["conv2d"].func, FUNC_TABLE["conv2d"].args
    scheduler = NaturalScheduler(type='random', batch_size=64)
    train_single_operator(matmul, args, scheduler, "llvm", CPU_MSG)
