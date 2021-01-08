import torch
from torchvision import transforms
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from datasets.PetsDetectionEncodbleDataset import PetsDetectionEncodableDataset
from datasets.ThorDepthEncodbleDataset import ThorDepthEncodableDataset
from datasets.NyuDepthEncodbleDataset import NyuDepthEncodableDataset
from datasets.NyuWalkableEncodbleDataset import NyuWalkableEncodableDataset
from datasets.TaskonomyInpaintingEncodbleDataset import TaskonomyInpaintingEncodableDataset
from datasets.EncodableDataset import EncodableDataset
from datasets.EncodableDataloader import EncodableDataloader
from models.PixelWisePredictionHead import PixelWisePredictionHead
from models.ResNet50Encoder import ResNet50Encoder
from models.VTABModel import VTABModel


if sys.argv[1] == "pets":
    dataset = PetsDetectionEncodableDataset(train=False)
    head = PixelWisePredictionHead(1)
    head.load_state_dict(torch.load(sys.argv[3], map_location=torch.device('cpu')))
elif sys.argv[1] == "thor-depth":
    dataset = ThorDepthEncodableDataset(train=False)
    head = PixelWisePredictionHead(1)
    head.load_state_dict(torch.load(sys.argv[3], map_location=torch.device('cpu')))
elif sys.argv[1] == "nyu-depth":
    dataset = NyuDepthEncodableDataset(train=False)
    head = PixelWisePredictionHead(1)
    head.load_state_dict(torch.load(sys.argv[3], map_location=torch.device('cpu')))
elif sys.argv[1] == "nyu-walkable":
    dataset = NyuWalkableEncodableDataset(train=False)
    head = PixelWisePredictionHead(1)
    head.load_state_dict(torch.load(sys.argv[3], map_location=torch.device('cpu')))
elif sys.argv[1] == "taskonomy-inpainting":
    dataset = TaskonomyInpaintingEncodableDataset(train=False)
    head = PixelWisePredictionHead(3)
    head.load_state_dict(torch.load(sys.argv[3], map_location=torch.device('cpu')))
else:
    print("Usage python visualize_mask_output.py "
          "<pets | nyu-walkable | nyu-depth | thor-depth | taskonomy-inpainting> "
          "<ENCODER_WEIGHTS_PATH> "
          "<TASK_HEAD_WEIGHTS_PATH>")
    exit()

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

model = VTABModel(ResNet50Encoder(weights=sys.argv[2]), head)

test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)


for img, label in test_dataloader:
    rgb_imgs = img.clone()
    with torch.no_grad():
        out = model(img)
        print(torch.nn.functional.l1_loss(out, label))
        if sys.argv[1] in ["pets", "nyu-walkable"]:
            out = torch.round(torch.sigmoid(out))
        if sys.argv[1] in ["taskonomy-inpainting"]:
            out = inv_normalize(out.detach()).numpy().transpose(0, 2, 3, 1)
            label = inv_normalize(label.detach()).numpy().transpose(0, 2, 3, 1)

    plt.figure(4)
    plt.imshow(out[4])
    # plt.imsave("inpainting-random.png", out[4] / np.max(out[4]))
    # plt.imshow(label[4])

    plt.figure(8)
    plt.imshow(out[8])
    # plt.imshow(label[8])

    plt.show()
    exit()
    fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(3, 10))
    axs[0, 0].set_title('Prediction')
    axs[0, 0].imshow(out[0])
    axs[0, 1].set_title('Label')
    axs[0, 1].imshow(label[0])
    axs[0, 2].set_title('Image')
    axs[0, 2].imshow(rgb_imgs[0].detach().numpy().transpose((1, 2, 0)))
    print(np.mean(np.abs(out - label)))
    print(np.min(out), np.max(out))
    print(np.min(label), np.max(label))
    for i in range(1, 5):
        axs[i, 0].imshow(out[i])
        axs[i, 1].imshow(label[i])
        axs[i, 2].imshow(rgb_imgs[i].detach().numpy().transpose((1, 2, 0)))
    plt.show()
    break
