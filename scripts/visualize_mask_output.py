import torch
import sys
import numpy as np
import matplotlib.pyplot as plt

from datasets.PetsEncodbleDataset import PetsEncodableDataset
from datasets.EncodableDataset import EncodableDataset
from datasets.EncodableDataloader import EncodableDataloader
from models.PixelWisePredictionHead import PixelWisePredictionHead
from models.ResNet50Encoder import ResNet50Encoder
from models.VTABModel import import VTABRunner


if sys.argv[1] == "pets":
    dataset = PetsEncodableDataset(train=False)
else:
    print("Usage python visualize_mask_output.py <pets | walkable | depth> <MODEL_NAME>")

test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

model =


for img, label in test_dataloader:
    test_dataloader = EncodableDataloader(
        test_dataloader,
        model,
        "Encoding Test Set",
        None,
        batch_size=10,
        shuffle=False,
        device="cpu",
        principal_directions=model.get_principal_directions()
    )
