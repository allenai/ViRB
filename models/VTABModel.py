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
        self.principal_directions = None

    def forward(self, x):
        x = self.encoder_forward(x)
        x = self.head_forward(x)
        return x

    def encoder_forward(self, x):
        if self.train_encoder:
            x = self.encoder(x)
            if hasattr(self.task_head, "pca_embedding_sizes") and self.task_head.pca_embedding_sizes() is not None:
                x = self.compress_encoding(x)
            return x
        with torch.no_grad():
            x = self.encoder(x)
            if hasattr(self.task_head, "pca_embedding_sizes") and self.task_head.pca_embedding_sizes() is not None:
                x = self.compress_encoding(x)
            return x

    def head_forward(self, x):
        return self.task_head(x)

    def compress_encoding(self, encoding):
        # Run PCA
        out = {}
        for name in self.task_head.pca_embedding_sizes():
            x = encoding[name].detach()
            if self.principal_directions is None:
                self.principal_directions = {}
            if name not in self.principal_directions:
                self.principal_directions[name] = self.get_principal_directions(
                    x, self.task_head.pca_embedding_sizes()[name]
                )
            out[name] = self.get_principal_components(x, self.principal_directions[name])
        return out

    def required_encoding(self):
        return self.task_head.required_encoding()

    def state_dict(self):
        sd = super().state_dict()
        if self.principal_directions is not None:
            for name, pd in self.principal_directions.items():
                sd["principal_directions."+name] = pd
        return sd

    def load_state_dict(self, state_dict, strict=False):
        if hasattr(self.task_head, "pca_embedding_sizes") and self.task_head.pca_embedding_sizes() is not None:
            pd_names = self.task_head.pca_embedding_sizes()
            self.principal_directions = {}
            for name in pr_names:
                self.principal_directions[name] = d["principal_directions."+name]
        super().load_state_dict(state_dict, strict)

    def get_principal_directions(self, x, num_dims):
        q = min(x.size(1), 1024)
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(-1, x.size(3))
        _, _, V = torch.pca_lowrank(x, q=q, niter=10)
        return V[:, :num_dims]

    def get_principal_components(self, x, V):
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)
            original_shape = x.shape
            x = x.reshape(-1, x.size(3))
        res = torch.matmul(x, V)
        if len(original_shape) == 4:
            res = res.reshape(original_shape[0], original_shape[1], original_shape[2], -1)
            res = res.permute(0, 3, 1, 2)
        return res
