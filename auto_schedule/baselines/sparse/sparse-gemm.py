import time
import numpy as np
import torch


def torch_spmm(M, N, K, dtype="float32", n_trial=1):
    spmm = torch.sparse.mm 
    # a_np = np.random.uniform(-0.91, 0.9, [M, K]).astype(dtype)
    # b_np = np.random.uniform(-0.91, 0.9, [K, N]).astype(dtype)
    # a_torch = torch.relu(torch.tensor(a_np)).to_sparse()
    # b_torch = torch.tensor(b_np)
    m = torch.distributions.bernoulli.Bernoulli(torch.tensor(0.9))
    a_torch = m.sample([M, K]).to_sparse()
    b_torch = m.sample([K, N])

    # warm-up
    res = spmm(a_torch, b_torch)
    beg = time.time()
    for i in range(n_trial):
        spmm(a_torch, b_torch)
    end = time.time()
    return (end - beg) * 1e3 / n_trial


def torch_spmv(M, K, dtype="float32", n_trial=1):
    spmm = torch.sparse.mm 
    # a_np = np.random.uniform(-0.91, 0.9, [M, K]).astype(dtype)
    # b_np = np.random.uniform(-0.91, 0.9, [K, 1]).astype(dtype)
    # a_torch = torch.relu(torch.tensor(a_np)).to_sparse()
    # b_torch = torch.tensor(b_np)
    m = torch.distributions.bernoulli.Bernoulli(torch.tensor(0.9))
    a_torch = m.sample([M, K]).to_sparse()
    b_torch = m.sample([K, 1])

    # warm-up
    res = spmm(a_torch, b_torch)
    beg = time.time()
    for i in range(n_trial):
        spmm(a_torch, b_torch)
    end = time.time()
    return (end - beg) * 1e3 / n_trial


if __name__ == "__main__":
    # for i in [1, 5, 10, 15, 20, 50]:
    #     size = i * 2**10
    #     try:
    #         res = torch_spmm(size, size, size, n_trial=10)
    #     except Exception as e:
    #         print(str(e))
    #         res = float("inf")
    #     print("Spmm pytorch: [scale: %d]: %f ms" % (size, res))
    for i in range(1, 10 + 1):
        size = i * 10 * 2**10
        try:
            res = torch_spmv(size, size, n_trial=10)
        except Exception as e:
            print(str(e))
            res = float("inf")
        print("Spmm pytorch: [scale: %d]: %f ms" % (size, res))
    