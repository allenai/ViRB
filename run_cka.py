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
import yaml

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

def run_cka(dataset):
    with open('configs/experiment_lists/default.yaml') as f:
        encoders = yaml.load(f)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ds = OmniDataset(dataset, max_imgs=10000)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=16)
    distances = {}
    for model_name, path in tqdm.tqdm(encoders.items()):
        model = ResNet50Encoder(path)
        model = model.to(device).eval()
        outs = []
        with torch.no_grad():
            for image in dl:
                image = image.to(device)
                out = model(image)
                outs.append((out["embedding"]).cpu().clone().numpy())
        outs = np.concatenate(outs, axis=0)
        distance = scipy.spatial.distance.pdist(outs, metric="cosine")
        distances[model_name] = distance
    keys = list(distances.keys())
    n = len(keys)
    heatmap = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x = distances[keys[i]]
            y = distances[keys[j]]
            heatmap[i, j] = heatmap[j, i] = 1 - scipy.spatial.distance.cosine(x, y)
    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(heatmap, annot=True)
    plt.title(dataset)
    ax.set_xticklabels(keys, rotation=30)
    ax.set_yticklabels(keys, rotation=0)
    plt.savefig("graphs/cka/%s" % dataset)
    plt.close()


def linear_cka(dataset):
    with open('configs/experiment_lists/default.yaml') as f:
        encoders = yaml.load(f)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ds = CalTech101EncodableDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=16)
    data = {}
    for model_name, path in tqdm.tqdm(encoders.items()):
        model = ResNet50Encoder(path)
        model = model.to(device).eval()
        outs = []
        with torch.no_grad():
            for image, _ in dl:
                image = image.to(device)
                out = model(image)
                outs.append(out["embedding"].cpu().half())
                break
            outs = torch.cat(outs, dim=0)
            # center columns
            outs -= outs.mean(dim=0)
            data[model_name] = outs
    keys = list(data.keys())
    n = len(keys)
    heatmap = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x = data[keys[i]].to(device)
            y = data[keys[j]].to(device)
            cka = (torch.norm(y.T @ x) ** 2) / (torch.norm(x.T @ x) * torch.norm(y.T @ y))
            heatmap[i, j] = heatmap[j, i] = cka
    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(heatmap, annot=True)
    plt.title(dataset)
    ax.set_xticklabels(keys, rotation=30)
    ax.set_yticklabels(keys, rotation=0)
    plt.savefig("graphs/cka/%s" % dataset)
    plt.close()


def main():
    linear_cka("Caltech")
    # run_cka("Thor")
    # run_cka("ImageNet")
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # ds = ImagenetEncodableDataset()
    # # ds = CalTech101EncodableDataset()
    # dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=16)
    # outs = []
    # model = ResNet50Encoder("pretrained_weights/SWAV_800.pt")
    # model = model.to(device).eval()
    # enc_start = time.time()
    # with torch.no_grad():
    #     for image, _ in tqdm.tqdm(dl):
    #         image = image.to(device)
    #         out = model(image)
    #         outs.append(out["embedding"].cpu().numpy())
    # enc_end = time.time()
    # enc_time = enc_end - enc_start
    # print("Encoding Time: %02d:%02d" % (enc_time // 60, enc_time % 60))
    # outs = np.concatenate(outs, axis=0)
    # print("Outs Shape", outs.shape)
    # distance = scipy.spatial.distance.pdist(outs, metric="cosine")
    # dist_end = time.time()
    # dist_time = dist_end - enc_end
    # print("Distance Time: %02d:%02d" % (dist_time // 60, dist_time % 60))
    # print("Distance Shape", distance.shape)


if __name__ == '__main__':
    main()
