import torch
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

DATASETS = [
    CalTech101EncodableDataset,
    nuScenesActionPredictionDataset
]


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


def compute_cka(a, b):
    assert a.size(0) == b.size(0)
    cs = torch.nn.CosineSimilarity(dim=1)
    a = a.repeat(a.size(0), *([1]*len(a.shape)))
    b = b.repeat(b.size(0), *([1]*len(b.shape)))
    return torch.mean(cs(a, b))


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
model = ResNet50Encoder("pretrained_weights/SWAV_800.pt")
model = model.to(device)
cka_correlations = torch.zeros((2, 2)).to(device)
for i in range(2):
    for j in range(i, 2):
        ds1 = DATASETS[i]()
        dl1 = torch.utils.data.DataLoader(ds1, batch_size=10, shuffle=True, num_workers=0)
        ds2 = DATASETS[i]()
        dl2 = torch.utils.data.DataLoader(ds2, batch_size=10, shuffle=True, num_workers=0)
        outs1 = {
            "embedding": []
        }
        outs2 = {
            "embedding": []
        }
        with torch.no_grad():
            count1 = 0
            for data, label in dl1:
                count1 += 1
                if count1 > 100:
                    break
                out = model(data.to(device))
                outs1["embedding"].append(out["embedding"])
            for key in outs1:
                outs1[key] = torch.cat(outs1[key], dim=0)
            count2 = 0
            for data, label in dl1:
                count2 += 1
                if count2 > 100:
                    break
                out = model(data.to(device))
                outs2["embedding"].append(out["embedding"])
            for key in outs2:
                outs2[key] = torch.cat(outs2[key], dim=0)
            cos_similarity = compute_cka(outs1["embedding"], outs2["embedding"])
            cka_correlations[i, j] = cka_correlations[j, i] = cos_similarity.mean()
print(cka_correlations)
