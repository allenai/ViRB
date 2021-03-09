import torch
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml
import os

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


def sort_heatmap_by_keys(heatmap, keys, corner="Supervised"):
    mat = np.zeros_like(heatmap)
    new_keys = keys.copy()
    new_keys.sort(key=lambda k: heatmap[keys.index(corner), keys.index(k)], reverse=True)
    for i, x in enumerate(new_keys):
        for j, y in enumerate(new_keys):
            mat[i, j] = heatmap[keys.index(x), keys.index(y)]
    return new_keys, mat


def linear_cka(dataset):
    with open('configs/experiment_lists/default.yaml') as f:
        encoders = yaml.load(f)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ds = OmniDataset(dataset, max_imgs=1500000)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=16)
    data = {}
    for model_name, path in tqdm.tqdm(encoders.items()):
        model = ResNet50Encoder(path)
        model = model.to(device).eval()
        outs = []
        with torch.no_grad():
            for image in dl:
                image = image.to(device)
                out = model(image)
                outs.append(out["embedding"].cpu())
            outs = torch.cat(outs, dim=0)
            # center columns
            outs -= outs.mean(dim=0)
            data[model_name] = outs
    keys = list(data.keys())
    n = len(keys)
    heatmap = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x = data[keys[i]]
            y = data[keys[j]]
            cka = (torch.norm(y.T @ x) ** 2) / (torch.norm(x.T @ x) * torch.norm(y.T @ y))
            heatmap[i, j] = heatmap[j, i] = cka
    np.save("graphs/cka/%s" % dataset, heatmap)
    # new_keys, mat = sort_heatmap_by_keys(heatmap, keys, corner="Supervised")
    # plt.figure(figsize=(20, 15))
    # ax = sns.heatmap(mat, annot=True)
    # plt.title(dataset)
    # ax.set_xticklabels(new_keys, rotation=30)
    # ax.set_yticklabels(new_keys, rotation=0)
    # plt.savefig("graphs/cka/%s" % dataset)
    # plt.close()


# def fro_matmul(a, b, stride=1000, device="cpu"):
#     s = 0.0
#     a = a.to(device)
#     with torch.no_grad():
#         for i in tqdm.tqdm(range(0, b.shape[1], stride)):
#             s += torch.sum(torch.pow(a @ b[:, i:min(i+stride, b.shape[1])].to(device), 2)).cpu().numpy()
#     return np.sqrt(s)

def fro_matmul(a, b, stride=1000, device="cpu"):
    s = 0.0
    a = a.to(device)
    with torch.no_grad():
        for i in range(0, b.shape[1], stride):
            s += torch.sum(torch.pow(a @ b[:, i:min(i+stride, b.shape[1])].to(device), 2)).cpu().numpy()
    return np.sqrt(s)


def layer_wise_linear_cka(model_name, path, device):
    model = ResNet50Encoder(path)
    model = model.to(device).eval()
    data = {}
    for dataset in DATASETS:
        ds = OmniDataset(dataset, max_imgs=10)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=16)
        outs = {}
        with torch.no_grad():
            for image in dl:
                image = image.to(device)
                out = model(image)
                for k in out:
                    if k not in outs:
                        outs[k] = []
                    outs[k].append(out[k].flatten(start_dim=1).cpu())
            for k in outs:
                outs[k] = torch.cat(outs[k], dim=0)
                # center columns
                outs[k] -= outs[k].mean(dim=0)
            data[dataset] = outs

    # sns.set()
    # fig, axes = plt.subplots(3, 5, figsize=(20, 15))
    # fig.suptitle(model_name)
    for idx, (dataset_name, corr) in enumerate(data.items()):
        keys = list(corr.keys())
        n = len(keys)
        heatmap = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                print("Filling Square %d %d" % (i, j))
                x = corr[keys[i]]
                y = corr[keys[j]]
                # cka = (np.linalg.norm(y.T @ x) ** 2) / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))
                cka = (fro_matmul(y.T, x, device=device) ** 2) / (fro_matmul(x.T, x, device=device) * fro_matmul(y.T, y, device=device))
                heatmap[i, j] = heatmap[j, i] = cka
        os.makedirs("graphs/cka/layer_wise/%s/" % model_name, exist_ok=True)
        np.save("graphs/cka/layer_wise/%s/%s" % (model_name, dataset_name), heatmap)
        # sns.heatmap(heatmap, annot=False, ax=axes.flat[idx])
        # axes.flat[idx].set_xticklabels(keys, rotation=30)
        # axes.flat[idx].set_yticklabels(keys, rotation=0)
        # axes.flat[idx].set_title(dataset_name)
    # plt.savefig("graphs/cka/layer_wise/%s" % model_name)
    # plt.close()


def two_model_layer_wise_linear_cka(model_name_a, path_a, model_name_b, path_b, dataset, device):
    model_a = ResNet50Encoder(path_a)
    model_a = model_a.to(device).eval()
    model_a.eval()
    model_b = ResNet50Encoder(path_b)
    model_b = model_b.to(device).eval()
    model_b.eval()
    ds = OmniDataset(dataset, max_imgs=10, resize=(112, 112))
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, num_workers=16)
    outs_a = {}
    outs_b = {}
    with torch.no_grad():
        # Encode the dataset with encoder a
        for image in dl:
            image = image.to(device)
            out = model_a(image)
            for k in out:
                if k not in outs_a:
                    outs_a[k] = []
                outs_a[k].append(out[k].flatten(start_dim=1).cpu())
        for k in outs_a:
            outs_a[k] = torch.cat(outs_a[k], dim=0)
            # center columns
            outs_a[k] -= outs_a[k].mean(dim=0)
        # Encode the dataset with encoder b
        for image in dl:
            image = image.to(device)
            out = model_b(image)
            for k in out:
                if k not in outs_b:
                    outs_b[k] = []
                outs_b[k].append(out[k].flatten(start_dim=1).cpu())
        for k in outs_b:
            outs_b[k] = torch.cat(outs_b[k], dim=0)
            # center columns
            outs_b[k] -= outs_b[k].mean(dim=0)
    # Clear memry
    del image
    del out
    del model_a
    del model_b
    torch.cuda.empty_cache()

    keys = list(outs_a.keys())
    n = len(keys)
    heatmap = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x = outs_a[keys[i]]
            y = outs_b[keys[j]]
            # cka = (torch.norm(y.T @ x) ** 2) / (torch.norm(x.T @ x) * torch.norm(y.T @ y))
            cka = (fro_matmul(y.T, x, device=device) ** 2) / (fro_matmul(x.T, x, device=device) * fro_matmul(y.T, y, device=device))
            heatmap[i, j] = heatmap[j, i] = cka
    os.makedirs("graphs/cka/multi_model_layer_wise/%s/" % dataset, exist_ok=True)
    np.save("graphs/cka/multi_model_layer_wise/%s/%s-%s" % (dataset, model_name_a, model_name_b), heatmap)


def main():
    # for dataset in DATASETS:
    #     linear_cka(dataset)
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

    with open('configs/experiment_lists/default.yaml') as f:
        encoders = yaml.load(f)
    argslist = []
    keys = list(encoders.keys())
    for i in range(len(encoders)):
        for j in range(i, len(encoders)):
            argslist.append((keys[i], encoders[keys[i]], keys[j], encoders[keys[j]]))
    nd = torch.cuda.device_count()
    if nd == 0:
        for args in tqdm.tqdm(argslist):
            two_model_layer_wise_linear_cka(*args, "ImageNet", "cpu")
    else:
        import threading
        count = 0
        while count < len(argslist):
            threads = []
            for gpu_id in range(min(len(argslist) - count, nd)):
                threads.append(
                    threading.Thread(
                        target=two_model_layer_wise_linear_cka,
                        args=(*argslist[count], "ImageNet", "cuda:%d" % gpu_id)
                    )
                )
                count += 1
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()


if __name__ == '__main__':
    main()
