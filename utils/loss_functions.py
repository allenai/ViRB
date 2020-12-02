import torch


def pixel_wise_prediction(out, labels):
    out = out.permute(0, 2, 3, 1).
    return None