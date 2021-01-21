import torch
import numpy as np


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
        if len(out.shape) == 4 and out.size(1) > 1:
            # layer_wise_label_mask = torch.zeros(
            #     [labels.size(0), torch.max(labels), labels.size(1), labels.size(2)],
            #     dtype=torch.long
            # )
            # layer_wise_label_mask[labels] = 1
            # layer_wise_label_mask = torch.stack([labels == x for x in range(labels.max() + 1)], dim=1).long()
            # prediction = torch.zeros_like(out)
            # prediction[torch.max(torch.softmax(out, dim=1), dim=1)] = 1
            ious = []
            _, prediction = torch.max(out[:, 1:, :, :], dim=1)
            mask = torch.round(torch.sigmoid(out[:, 1, :, :]))
            prediction *= mask
            for cat in torch.unique(labels):
                cat = cat.item()
                if cat == 0:
                    continue
                intersection = torch.logical_and(prediction == cat, labels == cat).sum(-1).sum(-1)
                union = torch.logical_or(prediction == cat, labels == cat).sum(-1).sum(-1)
                ious.append(torch.mean((intersection + 1e-8) / (union + 1e-8)).item())
            return np.mean(ious)

        else:
            layer_wise_label_mask = labels
            prediction = torch.round(torch.sigmoid(out))

            intersection = torch.logical_and(prediction, layer_wise_label_mask).sum(-1).sum(-1)
            union = torch.logical_or(prediction, layer_wise_label_mask).sum(-1).sum(-1)

            return torch.mean((intersection + 1e-8) / (union + 1e-8))
