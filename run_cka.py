import torch
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from models.ResNet50Encoder import ResNet50Encoder
from datasets.Caltech101EncldableDataset import CalTech101EncodableDataset
from datasets.Cifar100EncodbleDataset import CIFAR100EncodableDataset
from datasets.CityscapesSemanticSegmentationDataset import CityscapesSemanticSegmentationDataset
from datasets.CLEVERNumObjectsEncodbleDataset import CLEVERNumObjectsEncodableDataset
from datasets.dtdEncodbleDataset import dtdEncodableDataset
from datasets.EgoHandsDataset import EgoHandsDataset
from datasets.EurosatEncodbleDataset import EurosatEncodableDataset
from datasets.ImagenetEncodbleDataset import ImagenetEncodableDataset
from datasets.KineticsActionPrediction import KineticsActionPredictionDataset
from datasets.KITTIDataset import KITTIDataset
from datasets.nuScenesActionPredictionDataset import nuScenesActionPredictionDataset
from datasets.NyuDepthEncodbleDataset import NyuDepthEncodableDataset
from datasets.PetsEncodbleDataset import PetsEncodableDataset
from datasets.SUN397EncodbleDataset import SUN397EncodableDataset
from datasets.TaskonomyDepthEncodbleDataset import TaskonomyDepthEncodableDataset
from datasets.ThorActionPredictionDataset import ThorActionPredictionDataset


SEMANTIC_DATASETS = [
    CalTech101EncodableDataset,
    CIFAR100EncodableDataset,
    CityscapesSemanticSegmentationDataset,
    EgoHandsDataset,
    ImagenetEncodableDataset,
    PetsEncodableDataset,
    SUN397EncodableDataset,
    EurosatEncodableDataset,
    dtdEncodableDataset
]

STRUCTURAL_DATASETS = [
    CLEVERNumObjectsEncodableDataset,
    KineticsActionPredictionDataset,
    KITTIDataset,
    nuScenesActionPredictionDataset,
    NyuDepthEncodableDataset,
    TaskonomyDepthEncodableDataset,
    ThorActionPredictionDataset
]

DATASETS = SEMANTIC_DATASETS + STRUCTURAL_DATASETS


def flatten_model_by_layer(model):
    weights = {
        "block1": [],
        "block2": [],
        "block3": [],
        "block4": [],
    }
    for name, param in model.state_dict().items():
        if ("weight" in name or "bias" in name) and "bn" not in name:
            if "layer1" in name:
                weights["block1"].append(param.view(-1))
            elif "layer2" in name:
                weights["block2"].append(param.view(-1))
            elif "layer3" in name:
                weights["block3"].append(param.view(-1))
            elif "layer4" in name:
                weights["block4"].append(param.view(-1))
    for n, w in weights.items():
        weights[n] = torch.cat(w, dim=0)
    return weights


def compute_cka(x):
    with torch.no_grad():
        size = x.size(0)
        cs = torch.nn.CosineSimilarity(dim=1)
        a = []
        b = []
        for i in range(size-1):
            a.append(x[i].repeat(size-i-1, 1))
            b.append(x[i+1:])
        a = torch.cat(a, dim=0)
        b = torch.cat(b, dim=0)
        return cs(a, b)


######## Graph the norms of the model weights
# norm_stats = []
# for model in glob.glob("pretrained_weights/*.pt"):
#     weights = flatten_model_by_layer(ResNet50Encoder(model))
#     if "MoCo" in model:
#         method = "MoCo"
#     elif "SWAV" in model:
#         method = "SWAV"
#     elif "PIRL" in model:
#         method = "PIRL"
#     elif "SimCLR" in model:
#         method = "SimCLR"
#     elif "Supervised" in model:
#         method = "Supervised"
#     else:
#         method = "Random"
#     for n, w in weights.items():
#         norm_stats.append({
#             "encoder": model.split("/")[-1].split(".")[0],
#             "method": method,
#             "layer": n,
#             "norm": torch.linalg.norm(w, ord=1).item(),
#         })
#
# data = pd.DataFrame(norm_stats)
# data = data.sort_values('norm')
# sns.swarmplot(x="layer", y="norm", hue="method", data=data)
# plt.show()
#
# weights = flatten_model_by_layer(ResNet50Encoder("pretrained_weights/SWAV_800.pt"))
# data = pd.DataFrame([{"name":n, "norm":torch.linalg.norm(w, ord=1).item()} for n, w in weights.items()])
# sns.barplot(x="name", y="norm", data=data)
# plt.show()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_similarity = {}
for weights in glob.glob("pretrained_weights/*.pt"):
    # Set random seed
    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    model = ResNet50Encoder(weights)
    model = model.to(device)
    outs = {
        "embedding": []
    }
    for dataset in DATASETS:
        ds = dataset()
        dl = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)
        with torch.no_grad():
            count = 0
            for data, label in dl:
                count += 1
                if count > 10:
                    break
                out = model(data.to(device))
                outs["embedding"].append(out["embedding"])
    for key in outs:
        outs[key] = torch.cat(outs[key], dim=0)
    cka = compute_cka(outs["embedding"])
    print("Processing", weights.split("/")[1].split(".")[0], torch.linalg.norm(outs["embedding"], ord=1).item())
    model_similarity[weights.split("/")[1].split(".")[0]] = cka

cka_table = torch.ones((len(model_similarity), len(model_similarity)))
cs = torch.nn.CosineSimilarity(dim=0)
for i in range(len(model_similarity)):
    for j in range(i, len(model_similarity)):
        cka_table[i, j] = cka_table[j, i] = cs(
            model_similarity[list(model_similarity.keys())[i]], model_similarity[list(model_similarity.keys())[j]]
        )
np.save("cka_table.npy", cka_table.detach().cpu().numpy())
