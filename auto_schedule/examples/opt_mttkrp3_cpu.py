import time
import os
import torch
import argparse
import numpy as np
from auto_schedule.examples import FUNC_TABLE
from auto_schedule.test import test_graph_schedule_cpu_general_dx
from auto_schedule.train import Entity, train_op_schedule_cpu_general_dx


def run(M, N, K, L, model_path, epoch=5, sample_size=16, number=100, test=False):
    entities = []
    func = FUNC_TABLE["dense_MTTKRP2d"].func
    args = (M, N, K, L)
    entities.append(Entity("dense_MTTKRP2d", args))
    model_path = os.path.abspath(model_path)
    if not test:
        beg = time.time()
        train_op_schedule_cpu_general_dx(entities, epoch, sample_size, model_path)
        end = time.time()
        print("{}({}):".format("dense_MTTKRP2d", args))
        print("train done! use {}ms".format((end - beg) * 1e3))
    test_graph_schedule_cpu_general_dx(func, args, model_path, number=number)


def numpy_baseline(M, N, K, L, number=100):
    # refer to https://github.com/mrdmnd/scikit-tensor
    def khatrirao(matrices, reverse=False):
        # Compute the Khatri-Rao product of all matrices in list "matrices".
        # If reverse is true, does the product in reverse order.
        matorder = range(len(matrices)) if not reverse else list(reversed(range(len(matrices))))

        # Error checking on matrices; compute number of rows in result.
        # N = number of columns (must be same for each input)
        N = matrices[0].shape[1]
        # Compute number of rows in resulting matrix
        # After the loop, M = number of rows in result.
        M = 1
        for i in matorder:
            if matrices[i].ndim != 2:
                raise ValueError("Each argument must be a matrix.")
            if N != (matrices[i].shape)[1]:
                raise ValueError("All matrices must have the same number of columns.")
            M *= (matrices[i].shape)[0]

        # Computation
        # Preallocate result.
        P = np.zeros((M, N))

        # n loops over all column indices
        for n in range(N):
            # ab = nth col of first matrix to consider
            ab = matrices[matorder[0]][:, n]
            # loop through matrices
            for i in matorder[1:]:
                # Compute outer product of nth columns
                ab = np.outer(matrices[i][:, n], ab[:])
            # Fill nth column of P with flattened result
            P[:, n] = ab.flatten()
        return P

    def mttkrp(X, U, n):
        # Compute the matrix product of the n-mode matricization of self with the KR
        # product of all entries in U, a list of matrices, except the nth one.
        N = X.ndim
        order = list(range(N))
        order_sans_n = order[:]
        order_sans_n.remove(n)
        Xn = X.transpose([n] + order_sans_n)
        Xn = Xn.reshape(X.shape[n], X.size // X.shape[n])
        Z = khatrirao([U[i] for i in order_sans_n], reverse=True)
        return np.dot(Xn, Z)

    X = np.random.random((M, K, L))
    A = np.random.random((K, N))
    B = np.random.random((L, N))
    C = np.random.random((M, N))
    U = [C, A, B]
    beg = time.time()
    for i in range(number):
        D = mttkrp(X, U, 0)
    end = time.time()
    cost = (end - beg) * 1e3 / number
    return cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="train the model", action="store_true")
    parser.add_argument("--numpy", help="run numpy baseline", action="store_true")
    parser.add_argument("-a", "--auto_schedule", help="run auto-scheduler", action="store_true")
    parser.add_argument("-n", "--number", help="number of tests", type=int, default=100)
    parser.add_argument("-f", "--model_file_path", type=str, default="../logs/test_model.pkl")
    parser.add_argument("--params", help="M,N,K,L", type=str, default="32,32,32,32")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--sample", type=int, default=16)
    args = parser.parse_args()
    test = not args.train
    use_numpy = args.numpy
    use_auto = args.auto_schedule
    try:
        params = [int(x) for x in args.params.split(",")]
    except Exception as e:
        raise ValueError("Bad parameters, please refer to usage")
    M, N, K, L = params
    if use_numpy:
        cost = numpy_baseline(M, N, K, L, args.number)
    if use_auto:
        run(M, N, K, L, args.model_file_path, args.epoch, args.sample, args.number, test)

    arg_lst = [
        (1024, 1024, 8, 8),
        (32, 32, 32, 32),
        (256, 256, 16, 32)
    ]

    names = [
        "1024_1024_8_8",
        "32_32_32_32",
        "256_256_16_32"
    ]

    for i in range(len(arg_lst)):
        model_path = "../models/opt_mttkrp3_" + names[i] + "_cpu_process.pkl"
        entities = []
        func = FUNC_TABLE["dense_MTTKRP3"].func
        args = arg_lst[i]
        entities.append(Entity("dense_MTTKRP3", args))
        model_path = os.path.abspath(model_path)
        train_op_schedule_cpu_general_dx(entities, 5, 32, model_path, logfile="process_mttkrp_" + names[i] + "_cpu.txt", device="cuda:2")
