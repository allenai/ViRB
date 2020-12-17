import torch


def classification_error(out, labels):
    with torch.no_grad():
        predictions = torch.argmax(out, dim=1)
        return (predictions == labels).sum() / labels.size(0)


def binary_pixel_wise_prediction_loss(out, labels):
    with torch.no_grad():
        predictions = torch.round(torch.sigmoid(out))
        return (predictions == labels).sum() / (labels.size(0) * labels.size(2) * labels.size(3))


def iou(out, labels):
    with torch.no_grad():
        if labels.size(1) > 1:
            layer_wise_label_mask = torch.zeros([labels.size(0), torch.max(labels), labels.size(1), labels.size(2)])
            layer_wise_label_mask[out] = 1
        else:
            layer_wise_label_mask = labels

        out = torch.round(torch.sigmoid(out))

        intersection = (out == layer_wise_label_mask).sum(-1).sum(-1)
        union = (out + layer_wise_label_mask).sum(-1).sum(-1)

        return torch.mean(intersection / union)
