import torch.nn as nn


class MultiEmbeddingClassificationHead(nn.Module):

    def __init__(self, embedding_size, output_size, num_embeddings):
        super().__init__()
        self.embedding_size = embedding_size
        self.head = nn.Sequential(
            nn.Linear(embedding_size * num_embeddings, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_size)
        )
        # self.head = nn.Linear(embedding_size * num_embeddings, output_size)

    def forward(self, x):
        x = x["embedding"].float()
        x = x.view(x.size(0), -1)
        return self.head(x)

    def required_encoding(self):
        return ["embedding"]
