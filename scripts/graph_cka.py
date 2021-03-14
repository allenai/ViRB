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
#
# heatmaps = {}
# names = set()
# for file in glob.glob("../graphs/cka/multi_model_layer_wise/ImageNet/*.npy"):
#     name = file.split("/")[-1].replace("_200", "ImageNet").replace(".npy", "")
#     heatmaps[name] = np.load(file)
#     names.add(name.split("-")[0])
#     names.add(name.split("-")[1])
# names = list(names)
# names.sort()
# # names = ['SWAVImageNet', 'SWAVPlaces', 'SWAVCombination', 'SWAVTaskonomy', 'SWAVKinetics']
# names = ['MoCov2ImageNet', 'MoCov2Places', 'MoCov2Combination', 'MoCov2Taskonomy', 'MoCov2Kinetics']
# sns.set()
# n = len(names)
# fig, axes = plt.subplots(n, n, figsize=(15, 10))
# fig.suptitle("MoCov2 Layer by Layer Different Datasets")
# print(names)
# for i in range(n):
#     for j in range(n):
#         if "-".join((names[i], names[j])) in heatmaps:
#             hm = heatmaps["-".join((names[i], names[j]))]
#             sns.heatmap(hm, ax=axes[i][j], vmin=0, vmax=1)
#         elif "-".join((names[j], names[i])) in heatmaps:
#             hm = heatmaps["-".join((names[j], names[i]))]
#             sns.heatmap(hm, ax=axes[i][j], vmin=0, vmax=1)
#         if i == n-1:
#             axes[i][j].set_xlabel(names[j])
#         if j == 0:
#             axes[i][j].set_ylabel(names[i])
#         axes[i][j].get_xaxis().set_ticks([])
#         axes[i][j].get_yaxis().set_ticks([])
# plt.savefig("../graphs/cka/datasets/mocov2.pdf")
# plt.show()

# heatmaps = {}
# names = set()
# for file in glob.glob("../graphs/cka/multi_model_layer_wise/ImageNet/*.npy"):
#     name = file.split("/")[-1].replace(".npy", "")
#     if "MoCov2_" in name.split("-")[0] and "MoCov2_" in name.split("-")[1]:
#         heatmaps[name] = np.load(file)
# names = ["MoCov2_50", "MoCov2_100", "MoCov2_200"]
# sns.set()
# n = len(names)
# fig, axes = plt.subplots(n, n, figsize=(15, 10))
# fig.suptitle("SWAV Layer by Layer Different Datasets")
# print(names)
# for i in range(n):
#     for j in range(n):
#         if "-".join((names[i], names[j])) in heatmaps:
#             hm = heatmaps["-".join((names[i], names[j]))]
#             sns.heatmap(hm, ax=axes[i][j], vmin=0, vmax=1)
#         elif "-".join((names[j], names[i])) in heatmaps:
#             hm = heatmaps["-".join((names[j], names[i]))]
#             sns.heatmap(hm, ax=axes[i][j], vmin=0, vmax=1)
#         if i == n-1:
#             axes[i][j].set_xlabel(names[j])
#         if j == 0:
#             axes[i][j].set_ylabel(names[i])
#         axes[i][j].get_xaxis().set_ticks([])
#         axes[i][j].get_yaxis().set_ticks([])
# plt.savefig("../graphs/cka/num_training_steps/mocov2.pdf")
# plt.clf()

# heatmaps = {}
# names = set()
# for file in glob.glob("../graphs/cka/multi_model_layer_wise/ImageNet/*.npy"):
#     name = file.split("/")[-1].replace(".npy", "")
#     heatmaps[name] = np.load(file)
# names = ["SWAV_50", "SWAV_100", "SWAV_200"]
# sns.set()
# n = len(names)
# fig, axes = plt.subplots(n, n, figsize=(15, 10))
# fig.suptitle("SWAV Layer by Layer Different Datasets")
# print(names)
# for i in range(n):
#     for j in range(n):
#         if "-".join((names[i], names[j])) in heatmaps:
#             hm = heatmaps["-".join((names[i], names[j]))]
#             sns.heatmap(hm, ax=axes[i][j], vmin=0, vmax=1)
#         elif "-".join((names[j], names[i])) in heatmaps:
#             hm = heatmaps["-".join((names[j], names[i]))]
#             sns.heatmap(hm, ax=axes[i][j], vmin=0, vmax=1)
#         if i == n-1:
#             axes[i][j].set_xlabel(names[j])
#         if j == 0:
#             axes[i][j].set_ylabel(names[i])
#         axes[i][j].get_xaxis().set_ticks([])
#         axes[i][j].get_yaxis().set_ticks([])
# plt.savefig("../graphs/cka/num_training_steps/swav.pdf")
#
#
# heatmaps = {}
# names = set()
# for file in glob.glob("../graphs/cka/multi_model_layer_wise/ImageNet/*.npy"):
#     name = file.split("/")[-1].replace("_200", "ImageNet").replace(".npy", "")
#     heatmaps[name] = np.load(file)
#     names.add(name.split("-")[0])
#     names.add(name.split("-")[1])
# names = list(names)
# names.sort()
# # names = ['SWAVImageNet', 'SWAVPlaces', 'SWAVCombination', 'SWAVTaskonomy', 'SWAVKinetics']
# names = ['MoCov2ImageNet', 'MoCov2Places', 'MoCov2Combination', 'MoCov2Taskonomy', 'MoCov2Kinetics']
# sns.set()
# n = len(names)
# print(names)
# heatmap = np.zeros((n,n))
# for i in range(n):
#     for j in range(n):
#         if "-".join((names[i], names[j])) in heatmaps:
#             hm = heatmaps["-".join((names[i], names[j]))]
#         elif "-".join((names[j], names[i])) in heatmaps:
#             hm = heatmaps["-".join((names[j], names[i]))]
#         heatmap[i,j] = hm[0,-1]
# # plt.savefig("../graphs/cka/datasets/mocov2.pdf")
# plt.figure(1)
# plt.title("MoCo")
# sns.heatmap(heatmap, annot=True)
# print("Moco mean", np.mean(heatmap))
#
# names = ['SWAVImageNet', 'SWAVPlaces', 'SWAVCombination', 'SWAVTaskonomy', 'SWAVKinetics']
# n = len(names)
# print(names)
# heatmap = np.zeros((n,n))
# for i in range(n):
#     for j in range(n):
#         if "-".join((names[i], names[j])) in heatmaps:
#             hm = heatmaps["-".join((names[i], names[j]))]
#         elif "-".join((names[j], names[i])) in heatmaps:
#             hm = heatmaps["-".join((names[j], names[i]))]
#         heatmap[i,j] = hm[0,-1]
# # plt.savefig("../graphs/cka/datasets/mocov2.pdf")
# plt.figure(2)
# plt.title("SWAV")
# sns.heatmap(heatmap, annot=True)
# print("SWAV mean", np.mean(heatmap))

# heatmaps = []
# names = set()
# for file in glob.glob("../graphs/cka/multi_model_layer_wise/ImageNet/SWAV*.npy"):
#     name = file.split("/")[-1].replace("_200", "ImageNet").replace(".npy", "")
#     heatmaps.append(np.load(file))
# heatmap = np.mean(np.stack(heatmaps, axis=0), axis=0)
# plt.figure(1)
# plt.title("SWAV")
# sns.heatmap(heatmap, annot=True)
#
# heatmaps = []
# names = set()
# for file in glob.glob("../graphs/cka/multi_model_layer_wise/ImageNet/MoCov2*.npy"):
#     name = file.split("/")[-1].replace("_200", "ImageNet").replace(".npy", "")
#     heatmaps.append(np.load(file))
# heatmap = np.mean(np.stack(heatmaps, axis=0), axis=0)
# plt.figure(2)
# plt.title("MoCov2")
# sns.heatmap(heatmap, annot=True)
# plt.show()

heatmaps = {}
names = set()
for file in glob.glob("../graphs/cka/multi_model_layer_wise/ImageNet/*.npy"):
    name = file.split("/")[-1].replace("_200", "ImageNet").replace(".npy", "")
    heatmaps[name] = np.load(file)
fig, axes = plt.subplots(3, 3)
fig.suptitle("MoCo vs SWAV Same Dataset Different Method")
for i, corr in enumerate(['MoCov2ImageNet-MoCov2Combination', 'MoCov2ImageNet-MoCov2Places', 'MoCov2ImageNet-MoCov2ImageNet']):
    ax = sns.heatmap(heatmaps[corr], annot=True, vmin=0, vmax=1, ax=axes[0,i])
    ax.set_title(corr)
for i, corr in enumerate(['MoCov2Combination-SWAVCombination', 'MoCov2Places-SWAVPlaces', 'MoCov2ImageNet-SWAVImageNet']):
    ax = sns.heatmap(heatmaps[corr], annot=True, vmin=0, vmax=1, ax=axes[1,i])
    ax.set_title(corr)
for i, corr in enumerate(['SWAVImageNet-SWAVCombination', 'SWAVImageNet-SWAVPlaces', 'SWAVImageNet-SWAVImageNet']):
    ax = sns.heatmap(heatmaps[corr], annot=True, vmin=0, vmax=1, ax=axes[2,i])
    ax.set_title(corr)
plt.show()
