import torch
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from models.ResNet50Encoder import ResNet50Encoder
from datasets.OmniDataset import OmniDataset


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
    with torch.no_grad():
        sizea, sizeb = a.size(0), b.size(0)
        assert sizea == sizeb
        cs = torch.nn.CosineSimilarity(dim=1)
        aa = []
        bb = []
        for i in range(sizea-1):
            aa.append(a[i].repeat(sizea-i-1, 1).half())
            bb.append(b[i+1:].half())
        aa = torch.cat(aa, dim=0)
        bb = torch.cat(bb, dim=0)
        return cs(aa, bb)


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

DATASETS = [
    'Caltech', 'Cityscapes', 'CLEVR', 'dtd', 'Egohands', 'Eurosat',
    'ImageNet', 'Kinetics', 'nuScenes', 'NYU', 'Pets',
    'SUN397', 'Taskonomy', 'ThorActionPrediction'
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_similarity = {}
for weights in glob.glob("pretrained_weights/*.pt"):
    weights = "pretrained_weights/SWAV_800.pt"
    print("Processing", weights.split("/")[1].split(".")[0])
    model = ResNet50Encoder(weights)
    model = model.to(device)
    outs = {
        "embedding": []
    }
    ds = OmniDataset(DATASETS)
    dl = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)
    with torch.no_grad():
        for data in dl:
            out = model(data.to(device))
            outs["embedding"].append(out["embedding"])
    for key in outs:
        outs[key] = torch.cat(outs[key], dim=0)
    print("embeddings shape:", outs["embedding"].shape)
    cka = torch.zeros((15, 15))
    for i in range(15):
        for j in range(i, 15):
            cka[i, j] = cka[j, i] = torch.mean(compute_cka(
                outs["embedding"][i*1000:(i+1)*1000],
                outs["embedding"][j*1000:(j+1)*1000]
            ))
    np.save('SWAV_800_cka.numpy', cka.detach().cpu().numpy())


# cka_table = torch.ones((15, 15))
# cs = torch.nn.CosineSimilarity(dim=0)
# for i in range(len(model_similarity)):
#     for j in range(i, len(model_similarity)):
#         cka_table[i, j] = cka_table[j, i] = cs(
#             model_similarity[list(model_similarity.keys())[i]], model_similarity[list(model_similarity.keys())[j]]
#         )
# np.save("cka_table.npy", cka_table.detach().cpu().numpy())

# cka_table = np.load("cka_table.npy")
# sns.heatmap(cka_table)
# names = [weights.split("/")[1].split(".")[0] for weights in glob.glob("pretrained_weights/*.pt")]
# plt.xticks(np.arange(len(names)), names, rotation='vertical')
# plt.yticks(np.arange(len(names)), names, rotation='horizontal')
# plt.show()
