import torch

from datasets.Cifar100EncodbleDataset import CIFAR100EncodableDataset
from datasets.Caltech101EncldableDataset import CalTech101EncodableDataset
from datasets.PetsEncodbleDataset import PetsEncodableDataset
from models.ResNet50Encoder import ResNet50Encoder
from models.ClassificationHead import ClassificationHead
from tasks.VTABTask import VTABTask
from utils.error_functions import classification_error


def run():

    train_set = CalTech101EncodableDataset(train=True)
    test_set = CalTech101EncodableDataset(train=False)
    head = ClassificationHead(2048, train_set.num_classes(), "pretrained_weights/SWAV_800.pt")
    head.train()
    optim = torch.optim.Adam(head.parameters(), lr=3e-4)
    caltech101 = VTABTask(name="CalTech-101",
                        encoder=None,
                        head=head,
                        train_set=train_set,
                        test_set=test_set,
                        loss=torch.nn.CrossEntropyLoss(),
                        error=classification_error,
                        optimizer=optim,
                        out_dir="out/SWAV_800/caltech101",
                        batch_size=512,
                        num_workers=12)
    caltech101.train(50)

    train_set = CIFAR100EncodableDataset(train=True)
    test_set = CIFAR100EncodableDataset(train=False)
    head = ClassificationHead(2048, train_set.num_classes())
    head.train()
    optim = torch.optim.Adam(head.parameters())
    cifar100 = VTABTask(name="CIFAR-100",
                        encoder=None,
                        head=head,
                        train_set=train_set,
                        test_set=test_set,
                        loss=torch.nn.functional.cross_entropy,
                        error=classification_error,
                        optimizer=optim,
                        out_dir="out/SWAV_800/cifar100",
                        batch_size=512,
                        num_workers=12)
    cifar100.train(10)

    train_set = PetsEncodableDataset(train=True)
    test_set = PetsEncodableDataset(train=False)
    head = ClassificationHead(2048, train_set.num_classes())
    head.train()
    optim = torch.optim.Adam(head.parameters())
    cifar100 = VTABTask(name="Pets",
                        encoder=None,
                        head=head,
                        train_set=train_set,
                        test_set=test_set,
                        loss=torch.nn.functional.cross_entropy,
                        error=classification_error,
                        optimizer=optim,
                        out_dir="out/SWAV_800/pets",
                        batch_size=512,
                        num_workers=12)
    cifar100.train(100)


if __name__ == '__main__':
    run()
