import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np


class Tiny10(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer9 = nn.AdaptiveAvgPool2d((1,1))
        self.layer10 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)
        l7 = self.layer7(l6)
        l8 = self.layer8(l7)
        l9 = self.layer9(l8)
        l10 = self.layer10(l9)
        return [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]

    def loss(self, out, label):
        return F.cross_entropy(out, label)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Tiny10()
    model.to(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=12)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=12)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)[-1]
            loss = model.loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / 2000))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(images)[-1]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    #
    # heatmap = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(i, n):
    #         x = distances[keys[i]]
    #         y = distances[keys[j]]
    #         heatmap[i, j] = heatmap[j, i] = 1 - scipy.spatial.distance.cosine(x, y)


if __name__ == '__main__':
    main()
