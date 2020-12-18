import torch
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from datasets.PetsEncodbleDataset import PetsEncodableDataset
from datasets.EncodableDataset import EncodableDataset
from datasets.EncodableDataloader import EncodableDataloader
from models.PixelWisePredictionHead import PixelWisePredictionHead
from models.ResNet50Encoder import ResNet50Encoder
from models.VTABModel import VTABModel


if sys.argv[1] == "pets":
    dataset = PetsEncodableDataset(train=False)
    head = PixelWisePredictionHead(1)
else:
    print("Usage python visualize_mask_output.py <pets | walkable | depth> <MODEL_NAME>")
    exit()

model = VTABModel(ResNet50Encoder(), head)
model.load_state_dict(torch.load(sys.argv[2], map_location="cpu"))

with open(sys.argv[3]) as f:
    principle_directions = pickle.load(f)

test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
test_dataloader = EncodableDataloader(
    test_dataloader,
    model,
    "Encoding Test Set",
    None,
    batch_size=10,
    shuffle=False,
    device="cpu",
    principal_directions=principle_directions
)

for img, label in test_dataloader:
    with torch.no_grad():
        out = model(img)
        out = torch.round(torch.sigmoid(out))
    plt.figure(0)
    plt.imshow(out[0, 0].detach().numpy())
    plt.figure(1)
    plt.imshow(label[0, 0].detach().numpy())
    plt.figure(2)
    plt.imshow(img[0].detach().numpy().transpose((1,2,0)))
