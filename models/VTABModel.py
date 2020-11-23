import torch.nn as nn
import torch


class VTABModel(nn.Module):

    def __init__(self, encoder, task_head, train_encoder=False):
        super().__init__()
        self.encoder = encoder
        if not train_encoder:
            self.encoder.eval()
        self.task_head = task_head
        self.train_encoder = train_encoder

    def forward(self, x):
        if self.train_encoder:
            return self.task_head(self.encoder(x))
        else:
            with torch.no_grad():
                x = self.encoder(x)
            return self.task_head(x)
