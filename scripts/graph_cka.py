import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import tqdm


# for root in tqdm.tqdm(glob.glob("../graphs/cka/layer_wise/*")):
#     sns.set()
#     root_name = root.split("/")[-1]
#     keys = ["img", "b1", "b2", "b3", "b4", "b5", "emb"]
#     fig, axes = plt.subplots(3, 5, figsize=(20, 15))
#     fig.suptitle(root.split("/")[-1])
#     for idx, file in enumerate(glob.glob(root + "/*.npy")):
#         heatmap = np.load(file)
#         model_name = file.split("/")[-1].split(".")[0]
#         sns.heatmap(heatmap, annot=False, ax=axes.flat[idx], vmin=0, vmax=1)
#         axes.flat[idx].set_xticklabels(keys, rotation=30)
#         axes.flat[idx].set_yticklabels(keys, rotation=0)
#         axes.flat[idx].set_title(model_name)
#     plt.savefig("../graphs/cka/layer_wise/%s" % root_name)
#     plt.clf()

heatmaps = {}
names = set()
for file in glob.glob("../graphs/cka/multi_model_layer_wise/ImageNet/*.npy"):
    name = file.split("/")[-1].replace("_200", "ImageNet").replace(".npy", "")
    heatmaps[name] = np.load(file)
    names.add(name.split("-")[0])
    names.add(name.split("-")[1])
names = list(names)
names.sort()
sns.set()
n = len(names)
fig, axes = plt.subplots(n, n, figsize=(15, 10))
fig.suptitle("SWAV Layer by Layer Different Datasets")
print(names)
for i in range(n):
    for j in range(n):
        if "-".join((names[i], names[j])) in heatmaps:
            hm = heatmaps["-".join((names[i], names[j]))]
            sns.heatmap(hm, ax=axes[i][j], vmin=0, vmax=1)
        elif "-".join((names[j], names[i])) in heatmaps:
            hm = heatmaps["-".join((names[j], names[i]))]
            sns.heatmap(hm, ax=axes[i][j], vmin=0, vmax=1)
        if i == n-1:
            axes[i][j].set_xlabel(names[j])
        if j == 0:
            axes[i][j].set_ylabel(names[i])
        axes[i][j].get_xaxis().set_ticks([])
        axes[i][j].get_yaxis().set_ticks([])
plt.show()

