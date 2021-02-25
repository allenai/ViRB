import torch
import glob
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import tqdm
import time

from models.ResNet50Encoder import ResNet50Encoder
from datasets.OmniDataset import OmniDataset
from datasets.ImagenetEncodbleDataset import ImagenetEncodableDataset
from datasets.Caltech101EncldableDataset import CalTech101EncodableDataset


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
    'SUN397', 'Taskonomy', 'Thor'
]

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model_similarity = {}
# for weights in glob.glob("pretrained_weights/*.pt"):
#     print("Processing", weights.split("/")[1].split(".")[0])
#     model = ResNet50Encoder(weights)
#     model = model.to(device)
#     outs = {
#         "embedding": []
#     }
#     ds = OmniDataset(DATASETS)
#     dl = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)
#     with torch.no_grad():
#         for data in dl:
#             out = model(data.to(device))
#             outs["embedding"].append(out["embedding"])
#     for key in outs:
#         outs[key] = torch.cat(outs[key], dim=0)
#     cka = torch.zeros((14, 14, 499500))
#     for i in range(14):
#         for j in range(i, 14):
#             cka[i, j] = cka[j, i] = compute_cka(
#                 outs["embedding"][i*1000:(i+1)*1000],
#                 outs["embedding"][j*1000:(j+1)*1000]
#             )
#     np.save(weights.replace("pretrained_weights/", "").replace(".pt", ""), cka.detach().cpu().numpy())


# cka_table = torch.ones((15, 15))
# cs = torch.nn.CosineSimilarity(dim=0)
# for i in range(len(model_similarity)):
#     for j in range(i, len(model_similarity)):
#         cka_table[i, j] = cka_table[j, i] = cs(
#             model_similarity[list(model_similarity.keys())[i]], model_similarity[list(model_similarity.keys())[j]]
#         )
# np.save("cka_table.npy", cka_table.detach().cpu().numpy())
#
# for table in glob.glob("cka/*.npy"):
#     title = table.replace("cka/", "").replace(".npy", "")
#     print(title)
#     plt.figure(figsize=(10, 10))
#     plt.title(title)
#     cka_table = np.load(table)
#     cka_table = np.mean(cka_table, axis=2)
#     mindex = np.argmax(cka_table) // 14
#     results = [(DATASETS[i], cka_table[mindex, i]) for i in range(14)]
#     results.sort(key=lambda x: x[1], reverse=True)
#     results = [r[0] for r in results]
#     new_cka = np.zeros_like(cka_table)
#     for i, x in enumerate(results):
#         for j, y in enumerate(results):
#             new_cka[i, j] = cka_table[DATASETS.index(x), DATASETS.index(y)]
#     ax = sns.heatmap(new_cka)
#     ax.set_xticklabels(results, rotation=30)
#     ax.set_yticklabels(results, rotation=0)
#     plt.savefig("graphs/cka/" + title)
#     plt.clf()

# a = np.load("cka/SWAV_800.npy")
# b = np.load("cka/MoCov2_800.npy")

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ds = ImagenetEncodableDataset()
    # ds = CalTech101EncodableDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=10)
    outs = []
    model = ResNet50Encoder("pretrained_weights/SWAV_800.pt")
    model = model.to(device).eval()
    enc_start = time.time()
    with torch.no_grad():
        for image, _ in tqdm.tqdm(dl):
            image = image.to(device)
            out = model(image)
            outs.append(out["embedding"].cpu().numpy())
    enc_end = time.time()
    enc_time = enc_end - enc_start
    print("Encoding Time: %02d:%02d" % (enc_time // 60, enc_time % 60))
    outs = np.concatenate(outs, axis=0)
    print("Outs Shape", outs.shape)
    distance = scipy.spatial.distance.pdist(outs, metric="cosine")
    dist_end = time.time()
    dist_time = dist_end - enc_end
    print("Distance Time: %02d:%02d" % (dist_time // 60, dist_time % 60))
    print("Distance Shape", distance.shape)


if __name__ == '__main__':
    main()
