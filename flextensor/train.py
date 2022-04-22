import os
import argparse
import copy
import json
import torch
import numpy as np
from flextensor.model import WalkerGroup, PerformanceModel, rank_loss
from flextensor.task import TASK_TABLE, Task
from flextensor.scheduler import flatten_graph, Config
from flextensor.space import generate_space_intra_op, generate_space_inter_op


def train_perf(name, space):
    walker_group = WalkerGroup(name, space)
    walker_group.prepare_performance_data()
    walker_group.train_on_perf()


def train_q(name, space):
    walker_group = WalkerGroup(name, space)
    walker_group.load_walker_data()
    walker_group.train_walkers()


def train_for_schedule(task_key, q=False, perf=False):
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
    # train op
    for pos, op in enumerate(op_lst):
        if task.target == "cuda":
            space = generate_space_intra_op(op, down_graph, slevel=4)
        elif task.target == "llvm":
            space = generate_space_intra_op(op, down_graph, slevel=4, rlevel=3)
        else:
            raise RuntimeError("Currently no support for target %s"%task.target)
        name = task.category + "_op" + str(pos)
        if q:
            train_q(name, space)
        if perf:
            train_perf(name, space)        

    #################################################
    # train graph
    graph_space = generate_space_inter_op(op_lst, down_graph)
    name = task.category + "_graph"
    if q:
        train_q(name, graph_space)
    if perf:
        train_perf(name, graph_space)


def print_perf_data_info(data_path):
    with open(data_path, "r") as fin:
        can_out = False
        while not can_out:
            line = fin.readline()
            if len(line) <= 0:
                print("The data set has no valid data")
                can_out = True
            data = tuple(json.loads(line))
            if len(data[0]) > 0:
                print("Data input length:", len(data[0][0]))
                can_out = True


def query_perf_data_info(dataset):
    if len(dataset) <= 0:
        raise RuntimeError("Dataset size <= 0")
    else:
        data = dataset[0]
        if len(data[0]) <= 0:
            raise RuntimeError("In valid data")
        return len(data[0])


def normalize_perf_data(dataset):
    data_lst = []
    # normalize
    max_val = 0.0
    min_val = float("inf")
    for data in dataset[:]:
        for val in data[1]:
            if val != float("inf"):
                if val > max_val:
                    max_val = val
                elif val < min_val:
                    min_val = val
    interval = max(max_val - min_val, 1e-5)
    for data in dataset[:]:
        # filter empty data
        if len(data[0]) <= 0:
            continue
        # max_val = 0.0
        # min_val = float("inf")
        # for i in range(len(data[1])):
        #     if data[1][i] != float("inf"):
        #         if data[1][i] > max_val:
        #             max_val = data[1][i]
        #         elif data[1][i] < min_val:
        #             min_val = data[1][i]
        # interval = max(max_val - min_val, 1e-5)
        for i in range(len(data[1])):
            new_data = []
            new_data.append(copy.deepcopy(data[0][i]))
            new_data.append(copy.deepcopy(data[1][i]))
            if data[1][i] == float("inf"):
                new_data[1] = 1000.0
            # else:
            #     new_data[1] = (data[1][i] - min_val) / interval
            data_lst.append(new_data)
    return data_lst


def train_performance_model(data_path, model_path, epoch=10, batch_size=1, lr=0.02, override=False, train_ratio=0.8):
    # load data
    dataset = []
    with open(data_path, "r") as fin:
        for line in fin:
            data = tuple(json.loads(line))
            dataset.append(data)
    # prepare dataset
    dataset = normalize_perf_data(dataset)
    data_size = len(dataset)
    train_size = int(0.8 * data_size)
    test_size = data_size - train_size
    print("Train data size:", train_size, "test data size:", test_size)
    np.random.shuffle(dataset)
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    # get input length
    input_len = query_perf_data_info(dataset)
    print("Input length is:", input_len)

    # load model
    model = PerformanceModel(input_len)
    device = torch.device("cuda:0")
    model.to(device)
    # if os.path.exists(model_path) and not override and epoch > 0:
    #     raise RuntimeError("Existing model file %s" % model_path)
    # elif os.path.exists(model_path):
    #     print("Warning: override existing model file %s" % model_path)
    
    # train
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    print("Training begins...")
    for ep in range(epoch):
        model.train()
        print("Epoch", ep + 1, "begins:")
        np.random.shuffle(train_data)
        print("Train data shuffled")
        count_batch = 0
        acc_loss = 0.0
        inputs = []
        targets = []
        for count, data in enumerate(train_data):
            x, t = data
            inputs.append(x)
            targets.append(t)
            if (count + 1) % batch_size == 0:
                count_batch += 1
                inputs_torch = torch.FloatTensor(inputs).cuda()
                # targets_torch = torch.softmax(torch.FloatTensor(targets).cuda(), dim=-1)
                # ys = torch.softmax(model(inputs_torch).reshape(-1), dim=-1)
                targets_torch = torch.FloatTensor(targets).cuda()
                ys = model(inputs_torch).reshape(-1)
                # print("check ys=", ys.cpu().tolist())
                # print("check targets=", targets_torch.cpu().tolist())
                # print("check diff=", (ys -targets_torch).cpu().tolist())
                loss = torch.nn.functional.mse_loss(ys, targets_torch)
                # loss = rank_loss(ys, targets_torch)
                # loss = torch.nn.functional.binary_cross_entropy(ys, targets_torch)
                acc_loss = acc_loss + float(loss)
                optimizer.zero_grad()
                # for p in model.parameters():
                #     print("before", p.grad)
                loss.backward()
                # for p in model.parameters():
                #     print("after", p.grad)
                optimizer.step()
                # print("####| batch %d loss = %f" % (count_batch, float(loss)))
                # clear inputs and targets
                inputs = []
                targets = []
        if inputs and targets:
            # the remaining loss
            count_batch += 1
            inputs_torch = torch.FloatTensor(inputs).cuda()
            # targets_torch = torch.softmax(torch.FloatTensor(targets).cuda(), dim=-1)
            # ys = torch.softmax(model(inputs_torch).reshape(-1), dim=-1)
            targets_torch = torch.FloatTensor(targets).cuda()
            ys = model(inputs_torch).reshape(-1)
            # print("check ys=", ys.cpu().tolist())
            # print("check targets=", targets_torch.cpu().tolist())
            # print("check diff=", (ys -targets_torch).cpu().tolist())
            loss = torch.nn.functional.mse_loss(ys, targets_torch)
            # loss = rank_loss(ys, targets_torch)
            # loss = torch.nn.functional.binary_cross_entropy(ys, targets_torch)
            acc_loss = acc_loss + float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("####| the last batch %d loss = %f" % (count_batch, float(loss)))
        print("Accumulated loss of the whole epoch:", acc_loss)
        torch.save(model.state_dict(), model_path)
        print("Model saved")
        if (ep + 1) % 10 == 0:
            # validation
            print("Validation:")
            np.random.shuffle(test_data)
            print("Validation data shuffled")
            model.eval()
            hits = 0
            soft_hits = 0
            inputs = []
            targets = []
            count_batch = 0
            for count, data in enumerate(test_data):
                x, t = data
                inputs.append(x)
                targets.append(t)
                if (count + 1) % 2 == 0:
                    count_batch += 1
                    inputs_torch = torch.FloatTensor(inputs).cuda()
                    ys = model(inputs_torch)
                    ys_lst = list(enumerate(ys.cpu().tolist()))
                    targets_lst = list(enumerate(targets))
                    ys_sorted = sorted(ys_lst, key=lambda x: x[1])
                    targets_sorted = sorted(targets_lst, key=lambda x: x[1])
                    full_hit = True
                    soft_hit = True
                    # print(ys_sorted, targets_sorted)
                    for y_item, t_item in zip(ys_sorted, targets_sorted):
                        if y_item[0] != t_item[0]:
                            full_hit = False
                        elif abs(y_item[0] - t_item[0]) > 1:
                            soft_hit = False
                    if full_hit:
                        hits += 1
                    if soft_hit:
                        soft_hits += 1
                    inputs = []
                    targets = []

            print("Validation full accuracy:", float(hits) / count_batch * 100, "%")
            print("Validation soft accuracy:", float(soft_hits) / count_batch * 100, "%")
    print("Train done!")
    # test
    print("Testing begins...")
    np.random.shuffle(test_data)
    print("Test data shuffled")
    model.eval()
    hits = 0
    soft_hits = 0
    inputs = []
    targets = []
    count_batch = 0
    for count, data in enumerate(test_data):
        x, t = data
        inputs.append(x)
        targets.append(t)
        if (count + 1) % 2 == 0:
            count_batch += 1
            inputs_torch = torch.FloatTensor(inputs).cuda()
            ys = model(inputs_torch)
            ys_lst = list(enumerate(ys.cpu().tolist()))
            targets_lst = list(enumerate(targets))
            ys_sorted = sorted(ys_lst, key=lambda x: x[1])
            targets_sorted = sorted(targets_lst, key=lambda x: x[1])
            full_hit = True
            soft_hit = True
            for y_item, t_item in zip(ys_sorted, targets_sorted):
                if y_item[0] != t_item[0]:
                    full_hit = False
                elif abs(y_item[0] - t_item[0]) > 1:
                    soft_hit = False
            if full_hit:
                hits += 1
            if soft_hit:
                soft_hits += 1
            inputs = []
            targets = []

    print("Test full accuracy:", float(hits) / count_batch * 100, "%")
    print("Test soft accuracy:", float(soft_hits) / count_batch * 100, "%")
    print("Test done!")
    

# if __name__ == "__main__":
#     # this is somehow foolish because you need to create a particular task
#     task = Task(
#         "conv2d",
#         "yolo1", 
#         None, 
#         (1, 64, 112, 112, 192, 3, 1, 1, 1, 1), 
#         "cuda", 
#         2
#         )
#     train_for_schedule(task.key, q=False, perf=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", help="Type of training `perf` of `q`", type=str, default="perf")
    parser.add_argument("-d", "--data", help="Path to train data", type=str, default="")
    parser.add_argument("-m", "--model", help="Path to model", type=str, default="model.pkl")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--ratio", type=float, default=0.8)

    args = parser.parse_args()
    if args.type == "perf":
        train_performance_model(
            args.data, 
            args.model, 
            epoch=args.epoch, 
            batch_size=args.batch, 
            lr=args.lr, 
            override=args.override, 
            train_ratio=args.ratio
            )
    else:
        raise NotImplementedError("Currenly no support for type %s" % args.type)