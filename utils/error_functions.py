import torch


def classification_error(out, labels):
    with torch.no_grad():
        predictions = torch.argmax(out, dim=1)
        return (predictions == labels).sum() / labels.size(0)


def binary_pixel_wise_prediction_loss(out, labels):
    with torch.no_grad():
        predictions = torch.round(out)
        return (predictions == labels).sum() / labels.size(0)
