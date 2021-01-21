import torch
import torch.nn.functional as F


def weighted_l1_loss(out, labels):
    return torch.mean(torch.abs(out - labels) * labels)


def sparse_label_loss(out, label):
    prediction_loss = F.binary_cross_entropy_with_logits(out[:,0,:,:], (label != 0).float())
    prediction_mask = torch.round(torch.sigmoid(out[:,0,:,:]))
    prediction = out[:, 1:, :, :] * prediction_mask.unsqueeze(1)
    classification_loss = F.cross_entropy(prediction, label, ignore_index=0)
    return prediction_loss + classification_loss
