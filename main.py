import torch
import sys

from datasets.Cifar100EncodbleDataset import CIFAR100EncodableDataset
from datasets.Caltech101EncldableDataset import CalTech101EncodableDataset
from datasets.PetsEncodbleDataset import PetsEncodableDataset
from datasets.EurosatEncodbleDataset import EurosatEncodableDataset
from datasets.CLEVERNumObjectsEncodbleDataset import CLEVERNumObjectsEncodableDataset
from datasets.CLEVERDistEncodbleDataset import CLEVERDistEncodableDataset
from datasets.dtdEncodbleDataset import dtdEncodableDataset
from models.ResNet50Encoder import ResNet50Encoder
from models.ClassificationHead import ClassificationHead
from tasks.VTABTask import VTABTask
from utils.error_functions import classification_error


def run():

    train_set = CalTech101EncodableDataset(train=True)
    test_set = CalTech101EncodableDataset(train=False)
    head = ClassificationHead(2048, train_set.num_classes(), sys.argv[1])
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
                          out_dir="out/"+sys.argv[2]+"/caltech101",
                          batch_size=512,
                          num_workers=12)
    caltech101.train(100)

    train_set = CIFAR100EncodableDataset(train=True)
    test_set = CIFAR100EncodableDataset(train=False)
    head = ClassificationHead(2048, train_set.num_classes(), sys.argv[1])
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
                        out_dir="out/"+sys.argv[2]+"/cifar100",
                        batch_size=512,
                        num_workers=12)
    cifar100.train(25)

    train_set = PetsEncodableDataset(train=True)
    test_set = PetsEncodableDataset(train=False)
    head = ClassificationHead(2048, train_set.num_classes(), sys.argv[1])
    head.train()
    optim = torch.optim.Adam(head.parameters())
    pets = VTABTask(name="Pets",
                        encoder=None,
                        head=head,
                        train_set=train_set,
                        test_set=test_set,
                        loss=torch.nn.functional.cross_entropy,
                        error=classification_error,
                        optimizer=optim,
                        out_dir="out/"+sys.argv[2]+"/pets",
                        batch_size=512,
                        num_workers=12)
    pets.train(100)

    train_set = EurosatEncodableDataset(train=True)
    test_set = EurosatEncodableDataset(train=False)
    head = ClassificationHead(2048, train_set.num_classes(), sys.argv[1])
    head.train()
    optim = torch.optim.Adam(head.parameters())
    eurosat = VTABTask(name="Eurosat",
                        encoder=None,
                        head=head,
                        train_set=train_set,
                        test_set=test_set,
                        loss=torch.nn.functional.cross_entropy,
                        error=classification_error,
                        optimizer=optim,
                        out_dir="out/"+sys.argv[2]+"/eurosat",
                        batch_size=512,
                        num_workers=12)
    eurosat.train(50)

    train_set = dtdEncodableDataset(train=True)
    test_set = dtdEncodableDataset(train=False)
    head = ClassificationHead(2048, train_set.num_classes(), sys.argv[1])
    head.train()
    optim = torch.optim.Adam(head.parameters())
    dtd = VTABTask(name="dtd",
                        encoder=None,
                        head=head,
                        train_set=train_set,
                        test_set=test_set,
                        loss=torch.nn.functional.cross_entropy,
                        error=classification_error,
                        optimizer=optim,
                        out_dir="out/"+sys.argv[2]+"/dtd",
                        batch_size=512,
                        num_workers=12)
    dtd.train(100)

    train_set = CLEVERNumObjectsEncodableDataset(train=True)
    test_set = CLEVERNumObjectsEncodableDataset(train=False)
    head = ClassificationHead(2048, train_set.num_classes(), sys.argv[1])
    head.train()
    optim = torch.optim.Adam(head.parameters())
    CLEVERNumObjects = VTABTask(name="CLEVERNumObjects",
                        encoder=None,
                        head=head,
                        train_set=train_set,
                        test_set=test_set,
                        loss=torch.nn.functional.cross_entropy,
                        error=classification_error,
                        optimizer=optim,
                        out_dir="out/"+sys.argv[2]+"/CLEVERNumObjects",
                        batch_size=512,
                        num_workers=12)
    CLEVERNumObjects.train(20)

    train_set = CLEVERDistEncodableDataset(train=True)
    test_set = CLEVERDistEncodableDataset(train=False)
    head = ClassificationHead(2048, train_set.num_classes(), sys.argv[1])
    head.train()
    optim = torch.optim.Adam(head.parameters())
    CLEVERDist = VTABTask(name="CLEVERDist",
                        encoder=None,
                        head=head,
                        train_set=train_set,
                        test_set=test_set,
                        loss=torch.nn.functional.cross_entropy,
                        error=classification_error,
                        optimizer=optim,
                        out_dir="out/"+sys.argv[2]+"/CLEVERDist",
                        batch_size=512,
                        num_workers=12)
    CLEVERDist.train(20)


if __name__ == '__main__':
    run()
