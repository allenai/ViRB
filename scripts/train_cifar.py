import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm


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


class ResNet50Encoder(nn.Module):

    def __init__(self, weights=None):
        super().__init__()
        if weights:
            self.model = torchvision.models.resnet50(pretrained=False)
            weight_dict = torch.load(weights, map_location="cpu")
            self.load_state_dict(weight_dict, strict=False)
        else:
            self.model = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        res = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        res.append(x.clone())
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        res.append(x.clone())
        x = self.model.layer2(x)
        res.append(x.clone())
        x = self.model.layer3(x)
        res.append(x.clone())
        x = self.model.layer4(x)
        res.append(x.clone())
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        res.append(x.clone())

        return res


def fro_matmul(a, b, stride=1000, device="cpu"):
    s = 0.0
    a = a.to(device)
    with torch.no_grad():
        for i in range(0, b.shape[1], stride):
            s += torch.sum(torch.pow(a @ b[:, i:min(i+stride, b.shape[1])].to(device), 2)).cpu().numpy()
    return np.sqrt(s)


def train_cifar():
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
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    for epoch in range(100):
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
    return model


def run_cka(model, name, num_layers, im_size):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose(
        [transforms.Resize(im_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=12)
    model.to(device)
    model.eval()
    encodings = [[] for _ in range(num_layers)]
    print("Encoding")
    with torch.no_grad():
        for data in tqdm.tqdm(testloader):
            images, _ = data
            images = images.to(device)
            outputs = model(images)
            for i in range(num_layers):
                encodings[i].append(outputs[i].cpu())
    for i in range(len(encodings)):
        encodings[i] = torch.cat(encodings[i], dim=0).flatten(start_dim=1)
        encodings[i] = encodings[i] - encodings[i].mean()

    heatmap = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(i, num_layers):
            print("Computing block [%d][%d]" % (i,j))
            x = encodings[i]
            y = encodings[j]
            # cka = (torch.norm(y.T @ x) ** 2) / (torch.norm(x.T @ x) * torch.norm(y.T @ y))
            cka = (fro_matmul(y.T, x, device=device) ** 2) / (fro_matmul(x.T, x, device=device) * fro_matmul(y.T, y, device=device))
            heatmap[i, j] = heatmap[j, i] = cka.item()
    np.save(name, heatmap)


def show(name):
    sns.set()
    heatmap = np.load(name+".npy")
    sns.heatmap(heatmap)
    plt.title(name + " CIFAR 10 CKA")
    plt.savefig(name)
    plt.clf()


if __name__ == '__main__':
    model = ResNet50Encoder()
    run_cka(model, "resnet50", 6, (224, 224))
    model = train_cifar()
    run_cka(model, "tiny10", 10, (32, 32))
    show("resnet50")
    show("tiny10")
