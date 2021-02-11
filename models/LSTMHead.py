import torch.nn as nn


# class LSTMHead(nn.Module):
#
#     def __init__(self, num_classes):
#         super.__init__()
#         self.h0 = nn.Parameter(torch.randn(1, 20, 512).type(torch.FloatTensor), requires_grad=True)
#         self.rnn = nn.LSTM(2048, 512, 1)
#         self.classifier = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         x = self.rnn(x)
#         x = self.classifier(x)
#         return x

class LSTMHead(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(6 * 2048, 512),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.head(x)
        return x
