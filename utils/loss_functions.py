import torch


def weighted_l1_loss(out, labels):
    return torch.mean(torch.abs(out - labels) * labels)
