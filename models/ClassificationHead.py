import torch.nn as nn
import torch
import torchvision


class ClassificationHead(nn.Module):

    def __init__(self, embedding_size, output_size, encoder_weights):
        super().__init__()
        self.embedding_size = embedding_size
        resnet = torchvision.models.resnet50(pretrained=False)
        pretrainied_weights = torch.load(encoder_weights, map_location="cpu")
        pretrainied_weights = {k.replace("feature_extractor.resnet.", ""): v for k,v in pretrainied_weights.items()}
        resnet.load_state_dict(pretrainied_weights, strict=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.encoder.eval()
        self.head = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x).view(-1, self.embedding_size)
        return self.head(x)
