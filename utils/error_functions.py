import torch


def classification_error(out, labels):
    with torch.no_grad():
        predictions = torch.argmax(out)
        print(torch.count_nonzero(predictions == labels) / labels.size(0))
        return torch.count_nonzero(predictions == labels) / labels.size(0)
