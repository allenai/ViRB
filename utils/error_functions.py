import torch


def classification_error(out, labels):
    with torch.no_grad():
        _, predictions = torch.max(out, dim=1)
        return (predictions == labels).sum().item() / labels.size(0)
