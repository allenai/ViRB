import torch

from datasets.Cifar100EncodbleDataset import CIFAR100EncodableDataset
from datasets.Caltech101EncldableDataset import CalTech101EncodableDataset
from models.ResNet50Encoder import ResNet50Encoder
from models.ClassificationHead import ClassificationHead
from tasks.VTABTask import VTABTask
from utils.error_functions import classification_error


encoder = ResNet50Encoder()
encoder.eval()

train_set = CalTech101EncodableDataset(train=True)
test_set = CalTech101EncodableDataset(train=False)
head = ClassificationHead(2048, train_set.num_classes())
head.train()
optim = torch.optim.Adam(head.parameters(), lr=0.01)
# optim = torch.optim.SGD(head.parameters(), lr=0.01)
caltech100 = VTABTask(name="CalTech-101",
                    encoder=encoder,
                    head=head,
                    train_set=train_set,
                    test_set=test_set,
                    loss=torch.nn.CrossEntropyLoss(),
                    error=classification_error,
                    optimizer=optim,
                    out_dir="out/SWAV_800/CalTech-100",
                    batch_size=2048,
                    num_workers=12)
caltech100.train(100)


# train_set = CIFAR100EncodableDataset(train=True)
# test_set = CIFAR100EncodableDataset(train=False)
# head = ClassificationHead(2048, train_set.num_classes())
# head.train()
# optim = torch.optim.Adam(head.parameters())
# cifar100 = VTABTask(name="CIFAR-100",
#                     encoder=encoder,
#                     head=head,
#                     train_set=train_set,
#                     test_set=test_set,
#                     loss=torch.nn.functional.cross_entropy,
#                     error=classification_error,
#                     optimizer=optim,
#                     out_dir="out/SWAV_800/CIFAR-100",
#                     batch_size=256,
#                     num_workers=12)
# cifar100.train(100)
