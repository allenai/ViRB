import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_l1_loss(out, labels):
    return torch.mean(torch.abs(out - labels) * labels)


def sparse_label_loss(out, label):
    prediction_loss = F.binary_cross_entropy_with_logits(out[:,0,:,:], (label != 0).float())
    prediction_mask = torch.round(torch.sigmoid(out[:,0,:,:]))
    prediction = out * prediction_mask.unsqueeze(1)
    classification_loss = F.cross_entropy(prediction, label, ignore_index=0)
    return prediction_loss + classification_loss


def nonzero_l1_loss(out, label):
    return F.l1_loss(out[label != 0], label[label != 0])


class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
