import torch


def classification_error(out, labels):
    predictions = torch.argmax(out)
    return torch.nonzero(predictions == labels) / labels.size(0)
