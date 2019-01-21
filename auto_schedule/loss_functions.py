import torch
import torch.nn.functional as F
from functools import reduce


def ploss(logits, choice):
    """
    Return loss for action prediction

    :param logits:  torch.Tensor
        probabilities for an action
    :param choice:  int
    :return: torch.Tensor
        loss value
    """
    def _call(encourage=False):
        if encourage:
            loss = torch.exp(1 - logits[choice])
        else:
            loss = torch.exp(logits[choice])
        return loss
    return _call


def cross_entropy_loss(logits, choice):
    loss_fn = torch.nn.CrossEntropyLoss()
    sub_pos = 0
    sub_val = 0
    for i, v in enumerate(logits):
        if i == choice:
            continue
        if v > sub_val:
            sub_pos = i
            sub_val = v
    logits = torch.unsqueeze(logits, 0)

    def _call(encourage=False):
        if encourage:
            target = torch.LongTensor([choice])
            loss = loss_fn(logits, target)
        else:
            target = torch.LongTensor([sub_pos])
            loss = loss_fn(logits, target)
        return loss
    return _call


def rank_loss(logits, gamma=0.5):

    def _call(encourage=False):
        if encourage:
            loss = reduce(lambda x, y: gamma * x + y, logits)
        else:
            loss = reduce(lambda x, y: gamma * x + y, reversed(logits))
        return loss
    return _call


def kl_loss(output, target, align=0):
    align = max(len(output), len(target), align)
    if len(output) < align:
        output = torch.cat([output, torch.zeros(align-len(output))])
    if len(target) < align:
        target = torch.cat([target, torch.zeros(align-len(target))])
    output = F.log_softmax(output, dim=0)
    target = F.softmax(target, dim=0)
    loss = F.kl_div(output, target)
    return loss


def dev_loss(outputs):
    mean = torch.mean(outputs, dim=0)
    loss = torch.mean((outputs - mean)**2)
    return loss
