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
        x = self.encoder_forward(x)
        x = self.head_forward(x)
        return x

    def encoder_forward(self, x):
        if self.train_encoder:
            return self.encoder(x)
        with torch.no_grad():
            return self.encoder(x)

    def head_forward(self, x):
        return self.task_head(x)

    def required_encoding(self):
        return self.task_head.required_encoding()

    def pca_embeddings(self):
        if hasattr(self.task_head, "pca_embeddings"):
            return self.task_head.pca_embeddings()
        return None
