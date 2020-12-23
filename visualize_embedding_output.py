import torch
import sys
import pickle
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from datasets.EncodableDataloader import EncodableDataloader
from models.ClassificationHead import ClassificationHead
from models.ResNet50Encoder import ResNet50Encoder
from models.VTABModel import VTABModel


if sys.argv[1] == "Pets":
    from datasets.PetsEncodbleDataset import PetsEncodableDataset
    dataset = PetsEncodableDataset(train=False)
    head = ClassificationHead(2048, dataset.num_classes())
    head.load_state_dict(torch.load(sys.argv[3], map_location=torch.device('cpu')))
elif sys.argv[1] == "Sun":
    from datasets.SUN397EncodbleDataset import SUN397EncodableDataset
    dataset = SUN397EncodableDataset(train=False)
    head = ClassificationHead(2048, dataset.num_classes())
    head.load_state_dict(torch.load(sys.argv[3], map_location=torch.device('cpu')))
elif sys.argv[1] == "CIFAR100":
    from datasets.Cifar100EncodbleDataset import CIFAR100EncodableDataset
    dataset = CIFAR100EncodableDataset(train=False)
    head = ClassificationHead(2048, dataset.num_classes())
    head.load_state_dict(torch.load(sys.argv[3], map_location=torch.device('cpu')))
else:
    print("Usage python visualize_embedding_output.py "
          "<Pets | Sun | CIFAR100> "
          "<ENCODER_WEIGHTS_PATH> "
          "<TASK_HEAD_WEIGHTS_PATH> ")
    exit()

model = VTABModel(ResNet50Encoder(weights=sys.argv[2]), head)
model.eval()

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
for img, label in test_dataloader:
    rgb_imgs = img
    break

test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
test_dataloader = EncodableDataloader(
    test_dataloader,
    model,
    "Encoding Test Set",
    None,
    batch_size=100,
    shuffle=False,
    device="cpu",
    principal_directions=None
)
names = dataset.class_names()

for img, label in test_dataloader:
    with torch.no_grad():
        out = model.head_forward(img)
        out = torch.argmax(out, dim=1)
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(5, 5))
    # print(torch.mean(torch.abs(out - label)))
    # print(torch.min(out), torch.max(out))
    # print(torch.min(label), torch.max(label))
    for i in range(25):
        img = inv_tensor = inv_normalize(rgb_imgs[i].detach()).numpy().transpose((1, 2, 0))
        axs[i//5, i%5].imshow(img)
        axs[i//5, i%5].set_title('P:%s\nL:%s' % (names[out[i].item()], names[label[i].item()]))
    plt.show()
    break
