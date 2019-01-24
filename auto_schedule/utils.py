import numpy as np
import torch
from auto_schedule.config import UNSURE


def to_int(expr):
    try:
        res = int(expr)
    except Exception:
        res = UNSURE
    return res


def to_tuple(expr_tuple):
    return tuple([to_int(x) for x in expr_tuple])


def split_part_names(original, parts):
    assert isinstance(original, str) and isinstance(parts, int)
    ret = [original + "." + str(i) for i in range(parts)]
    return ret


def lower_bound(ary, beg, end, target):
    if beg >= end:
        return -1
    if beg == end - 1:
        if ary[beg] < target:
            return beg
        else:
            return -1
    if target <= ary[beg]:
        return -1
    while beg < end - 1:
        mid = (beg + end) // 2
        if ary[mid] < target:
            beg = mid
        else:
            end = mid
    return beg


def get_nearest_factor(x, factor):
    if not isinstance(x, int):
        raise ValueError("Only support int type, but get type:{}, value={}".format(type(x), x))
    factors = []
    s = int(np.sqrt(x))
    for i in range(1, s):
        if x % i == 0:
            factors.extend([i, x // i])
    if x % s == 0:
        factors.append(s)
    factors = list(sorted(factors))
    pos = lower_bound(factors, 0, len(factors), int(factor))
    if pos == -1:
        a, b = 1, x
    else:
        a, b = factors[pos], factors[min(len(factors)-1, pos + 1)]
    if factor / a < b / factor:
        return a
    else:
        return b


def slow_get_nearest_factor(x, factor):
    if not isinstance(x, int):
        raise ValueError("Only support int type, but get type:{}, value={}".format(type(x), x))
    a, b = int(factor), int(factor) + 1
    res_a, res_b = 1, x
    while 2 <= a <= x:
        if x % a == 0:
            res_a = a
            break
        a = a - 1
    while 2 <= b <= x:
        if x % b == 0:
            res_b = b
            break
        b = b + 1
    if res_b / factor < factor / res_a:
        res_a = res_b
    return res_a


def get_split(length, logits):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach()
        logits = logits.numpy()
    cur_len = length
    ret = []
    nparts = len(logits)
    exp_logits = np.exp(logits)
    cur_total = np.sum(exp_logits)
    for i in range(nparts-1):
        logit = exp_logits[i] / cur_total
        factor = get_nearest_factor(cur_len, np.exp(logit * np.log(cur_len)))
        cur_len = cur_len // factor
        cur_total -= exp_logits[i]
        ret.append(factor)
    ret.append(cur_len)
    return ret


def find_nearest_factor(number, target):
    if target >= number:
        return number
    if target < 1:
        return 1
    tmp = target
    a = 1
    while tmp >= 1:
        if number % tmp == 0:
            a = tmp
            break
        tmp -= 1
    tmp = target + 1
    b = number
    while tmp < number:
        if number % tmp == 0:
            b = tmp
            break
        tmp += 1
    if abs(b - target) < abs(a - target):
        return b
    return a
