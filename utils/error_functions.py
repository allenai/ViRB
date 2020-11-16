import torch


def classification_error(out, labels):
    with torch.no_grad():
        predictions = torch.argmax(out, dim=1)
        return torch.count_nonzero(predictions == labels) / labels.size(0)
