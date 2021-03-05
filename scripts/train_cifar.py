import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

def fro_matmul(a, b, stride=100, device="cpu"):
    s = 0.0
    a = a.to(device)
    b = b.to(device)
    with torch.no_grad():
        for i in range(0, b.shape[1], stride):
            s += torch.sum(torch.pow(a @ b[:, i:min(i+stride, b.shape[1])], 2)).cpu().numpy()
    return np.sqrt(s)

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
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)

    for epoch in range(350):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)[-1]
            loss = model.loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / 2000))
        scheduler.step()

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)[-1]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    model.eval()
    encodings = [[] for _ in range(10)]
    with torch.no_grad():
        for data in testloader:
            images, _ = data
            images = images.to(device)
            outputs = model(images)
            for i in range(10):
                encodings[i].append(outputs[i])
    for i in range(len(encodings)):
        encodings[i] = torch.cat(encodings[i], dim=0).flatten(start_dim=1)
        encodings[i] = encodings[i] - encodings[i].mean()

    heatmap = np.zeros((10, 10))
    for i in range(10):
        for j in range(i, 10):
            x = encodings[i]
            y = encodings[j]
            # cka = (torch.norm(y.T @ x) ** 2) / (torch.norm(x.T @ x) * torch.norm(y.T @ y))
            cka = (fro_matmul(y.T, x) ** 2) / (fro_matmul(x.T, x) * fro_matmul(y.T, y))
            heatmap[i, j] = heatmap[j, i] = cka.item()
    np.save("cifar_tiny10_layerwise_cka", heatmap)

def show():
    sns.set()
    heatmap = np.load("cifar_tiny10_layerwise_cka.npy")
    sns.heatmap(heatmap)
    plt.show()


if __name__ == '__main__':
    # show()
    main()
