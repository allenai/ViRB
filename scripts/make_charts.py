import json
import glob
import csv
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas
import scipy
import math

import scipy
import scipy.stats
import os

REAL_NAMES = {
    "Imagenet": "ImageNet Cls.",
    "Imagenetv2": "ImageNet v2 Cls.",
    "Pets": "Pets Cls.",
    "CIFAR-100": "CIFAR-100 Cls.",
    "CalTech-101": "CalTech Cls.",
    "Eurosat": "Eurosat Cls.",
    "dtd": "dtd Cls.",
    "SUN397": "SUN Scene Cls.",
    "KineticsActionPrediction": "Kinetics Action Pred.",
    "CLEVERNumObjects": "CLEVR Count",
    "THORNumSteps": "THOR Num. Steps",
    "THORActionPrediction": "THOR Egomotion",
    "nuScenesActionPrediction": "nuScenes Egomotion",
    "PetsDetection": "Pets Instance Seg.",
    "CityscapesSemanticSegmentation": "Cityscapes Seg.",
    "EgoHands": "EgoHands Seg.",
    "NYUDepth": "NYU Walkable",
    "NYUWalkable": "NYU Depth",
    "THORDepth": "THOR Depth",
    "TaskonomyDepth": "Taskonomy Depth",
    "KITTI": "KITTI Opt. Flow",
    "dtd-Deeplabv3": "dtd Cls. DeepLabv3",
    "Pets-Deeplabv3": "Pets Cls. DeepLabv3",
}


ALL_EXPERIMENTS = [
    'Random',
    'Supervised',
    'SWAV_800',
    'MoCov2_800',
    'MoCov1_200',
    'PIRL',
    'SimCLR_1000',
    'MoCov2_200',
    'MoCov2Combination',
    'MoCov2Taskonomy',
    'MoCov2Kinetics',
    'MoCov2Places',
    'MoCov2HalfImagenet',
    'MoCov2LogImagenet',
    'MoCov2UnbalancedImagenet',
    'MoCov2QuarterImagenet',
    'SWAV_200',
    # 'SWAV_200_2',
    'SWAVCombination',
    'SWAVTaskonomy',
    'SWAVKinetics',
    'SWAVPlaces',
    'SWAVHalfImagenet',
    'SWAVLogImagenet',
    'SWAVUnbalancedImagenet',
    'SWAVQuarterImagenet',
    'SWAVHalfImagenet_100',
    'MoCov2HalfImagenet_100',
    'SWAVUnbalancedImagenet_100',
    'MoCov2UnbalancedImagenet_100',
    'MoCov2_100',
    'MoCov2_50',
    'SWAV_50',
    # 'SWAV_100'
]

MOCOV2_EXPERIMENTS = [
    'MoCov2_800',
    'MoCov2_200',
    'MoCov2Combination',
    'MoCov2Taskonomy',
    'MoCov2Kinetics',
    'MoCov2Places',
    'MoCov2HalfImagenet',
    'MoCov2LogImagenet',
    'MoCov2QuarterImagenet',
    'MoCov2Imagenet_100'
]

MOCOV2_200_EXPERIMENT = [
    'MoCov2_200',
    'MoCov2Combination',
    'MoCov2Taskonomy',
    'MoCov2Kinetics',
    'MoCov2Places',
]

SWAV_EXPERIMENTS = [
    'SWAV_800',
    'SWAV_200',
    'SWAVCombination',
    'SWAVTaskonomy',
    'SWAVKinetics',
    'SWAVPlaces',
    'SWAVHalfImagenet',
    'SWAVLogImagenet',
    'SWAVUnbalancedImagenet',
    'SWAVQuarterImagenet'
]

SWAV_200_EXPERIMENTS = [
    'SWAV_200',
    'SWAVCombination',
    'SWAVTaskonomy',
    'SWAVKinetics',
    'SWAVPlaces',
]

IMAGENET_FULL_EXPERIMENTS = [
    'Random',
    'Supervised',
    'SWAV_800',
    'MoCov2_800',
    'MoCov1_200',
    'PIRL',
    'SimCLR_1000',
    'MoCov2_200',
    'SWAV_200'
]

HALF_IMAGENE_EXPERIMENTS = [
    'SWAVHalfImagenet',
    'MoCov2HalfImagenet',
]
UNBALANCED_IMAGENET_EXPERIMENTS = [
    'MoCov2UnbalancedImagenet',
    'SWAVUnbalancedImagenet',
]
IMAGENET_100_EPOCH_EXPERIMENTS = [
    'MoCov2_100'
]

QUARTER_IMAGENET_EXPERIMENTS = [
    'MoCov2QuarterImagenet',
    'SWAVQuarterImagenet'
]
LOG_IMAGENET_EXPERIMENTS = [
    'SWAVLogImagenet',
    'MoCov2LogImagenet',
]
HALF_IMAGENE_100_EXPERIMENTS = [
    'SWAVHalfImagenet_100',
    'MoCov2HalfImagenet_100',
]
UNBALANCED_IMAGENET_100_EXPERIMENTS = [
    'MoCov2UnbalancedImagenet_100',
    'SWAVUnbalancedImagenet_100',
]
IMAGENET_50_EPOCH_EXPERIMENTS = [
    'MoCov2_50',
    'SWAV_50'
]

PLACES_EXPERIMENTS = [
    'SWAVPlaces',
    'MoCov2Places',
]
KINETICS_EXPERIMENTS = [
    'MoCov2Kinetics',
    'SWAVKinetics',
]
TASKONOMY_EXPERIMENTS = [
    'MoCov2Taskonomy',
    'SWAVTaskonomy',
]
COMBO_EXPERIMENTS = [
    'MoCov2Combination',
    'SWAVCombination',
]

### TASKS
EMBEDDING_SEMANTIC_TASKS = [
    "Imagenet",
    "Imagenetv2",
    "Pets",
    "Pets-Deeplabv3",
    "CIFAR-100",
    "CalTech-101",
    "Eurosat",
    "dtd",
    "dtd-Deeplabv3",
    "SUN397",
    "KineticsActionPrediction",
]
EMBEDDING_STRUCTURAL_TASKS = [
    "CLEVERNumObjects",
    "THORNumSteps",
    "THORActionPrediction",
    "nuScenesActionPrediction"
]
PIXELWISE_SEMANTIC_TASKS = [
    "PetsDetection",
    "CityscapesSemanticSegmentation",
    "EgoHands"
]
PIXELWISE_STRUCTURAL_TASKS = [
    "NYUDepth",
    "NYUWalkable",
    "THORDepth",
    "TaskonomyDepth",
    "KITTI"
]

ALL_TASKS = EMBEDDING_SEMANTIC_TASKS + EMBEDDING_STRUCTURAL_TASKS + PIXELWISE_SEMANTIC_TASKS + PIXELWISE_STRUCTURAL_TASKS
EMBEDDING_TASKS = EMBEDDING_SEMANTIC_TASKS + EMBEDDING_STRUCTURAL_TASKS
PIXEL_TASKS = PIXELWISE_STRUCTURAL_TASKS + PIXELWISE_SEMANTIC_TASKS

ACC_TASKS = EMBEDDING_TASKS
MIOU_TASKS = PIXELWISE_SEMANTIC_TASKS + ["NYUWalkable"]
L1_TASKS = ["NYUDepth", "THORDepth", "TaskonomyDepth", "KITTI"]

REVERSED_SUCCESS_TASKS = [
    "TaskonomyInpainting",
    "TaskonomyEdges",
    "NYUDepth",
    "THORDepth",
    "TaskonomyDepth"
]


def get_best_result(experiments, run, include_names=False, c=1.0):
    res = []
    for e in experiments:
        datapoints = []
        datapoint_files = glob.glob("out/%s/%s-*/results.json" % (e, run))
        if len(datapoint_files) == 0:
            continue
        for f in datapoint_files:
            with open(f) as fp:
                datapoints.append(c * float(json.load(fp)["test_accuracy"]))
        if include_names:
            res.append((e, max(datapoints)))
        else:
            res.append(max(datapoints))
    return res


def get_all_results(experiments, run, c=1.0):
    res = {}
    for e in experiments:
        datapoints = []
        datapoint_files = glob.glob("out/%s/%s*/results.json" % (e, run))
        if len(datapoint_files) == 0:
            continue
        for f in datapoint_files:
            optimizer = f.replace("out/%s/%s-" % (e, run), "").replace("/results.json", "").split("-")
            with open(f) as fp:
                try:
                    datapoints.append({
                        "optimizer": optimizer[0],
                        "lr": float(optimizer[1]),
                        "result": c * float(json.load(fp)["test_accuracy"])
                    })
                except:
                    print("Problem with parsing")
        res[e] = datapoints
    return res


def get_normalized_summed_scores(data):
    embedding_matrix = np.zeros((len(EMBEDDING_TASKS), len(data["Encoder"])))
    for i, task in enumerate(EMBEDDING_TASKS):
        task_data = np.array(data[task])
        task_data -= np.min(task_data)
        task_data /= np.max(task_data)
        embedding_matrix[i] = task_data
    embedding_means = embedding_matrix.mean(axis=0)
    pixel_matrix = np.zeros((len(PIXEL_TASKS), len(data["Encoder"])))
    for i, task in enumerate(PIXEL_TASKS):
        task_data = np.array(data[task])
        if task in REVERSED_SUCCESS_TASKS:
            task_data *= -1.0
        task_data -= np.min(task_data)
        task_data /= np.max(task_data)
        pixel_matrix[i] = task_data
    pixel_means = pixel_matrix.mean(axis=0)
    return [
        {
            "Encoder": name,
            "Method": data["Method"][i],
            "Embedding": embedding_means[i],
            "Pixel": pixel_means[i]
        } for i, name in enumerate(data["Encoder"])
    ]



#### Taskonomy Inpainting number of channels
# RUNS = ["Random", "Supervised", "SWAV_800", "MoCov2_800", "PIRL", "MoCov2_200", "MoCov2Taskonomy", "SWAVTaskonomy"]
# end_to_end = get_best_result([r+"-end-to-end" for r in RUNS], "TaskonomyInpainting")
# plt.scatter(["End to End (1 hour)"]*len(end_to_end), end_to_end)
# full_channel = get_best_result([r+"-full-channel" for r in RUNS], "TaskonomyInpainting")
# plt.scatter(["Full Channel (53 minutes)"]*len(full_channel), full_channel)
# encoded_2_128 = get_best_result([r+"-2-128" for r in RUNS], "TaskonomyInpainting")
# plt.scatter(["2-128 (43 minutes)"]*len(encoded_2_128), encoded_2_128)
# encoded_4_128 = get_best_result([r+"-4-128" for r in RUNS], "TaskonomyInpainting")
# plt.scatter(["4-128 (45 minutes)"]*len(encoded_4_128), encoded_4_128)
# plt.show()







#### Converting the output to csv format
def make_csv():
    experiment_results = {name.replace("Imagenet", "IN"): {} for name in ALL_EXPERIMENTS}
    for task in ALL_TASKS:
        if task in REVERSED_SUCCESS_TASKS:
            res = get_best_result(ALL_EXPERIMENTS, task, include_names=True, c=-1.0)
        else:
            res = get_best_result(ALL_EXPERIMENTS, task, include_names=True)
        rankings, _ = zip(*sorted(res, key=lambda x: x[1], reverse=True))
        # print(task, len(res))
        mean = [r for n, r in res if n == "Supervised"][0]
        std = np.std([r for _, r in res])
        for name, number in res:
            sn = name.replace("Imagenet", "IN")
            experiment_results[sn][task] = number
            experiment_results[sn][task+"-rank"] = rankings.index(name)+1
            experiment_results[sn][task+"-normalized"] = (number - mean) / std

    with open('results.csv', mode='w') as csv_file:
        fieldnames = ["Encoder", "Method", "Dataset", "Balance", "Epochs", "Updates", "DatasetSize"] + ALL_TASKS + \
                     [task+"-normalized" for task in ALL_TASKS] + [task+"-rank" for task in ALL_TASKS]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for name, results in experiment_results.items():

            if "MoCo" in name:
                method = "MoCo"
            elif "SWAV" in name:
                method = "SWAV"
            elif "PIRL" in name:
                method = "PIRL"
            elif "SimCLR" in name:
                method = "SimCLR"
            elif "Supervised" in name:
                method = "Supervised"
            elif "Random" in name:
                method = "Random"
            else:
                method = "Other"

            if "Taskonomy" in name:
                dataset = "Taskonomy"
            elif "Places" in name:
                dataset = "Places"
            elif "Kinetics" in name:
                dataset = "Kinetics"
            elif "Combination" in name:
                dataset = "Combination"
            elif "Half" in name:
                dataset = "HalfImagenet"
            elif "Quarter" in name:
                dataset = "QuarterImagenet"
            elif "Unbalanced" in name:
                dataset = "UnbalancedImagenet"
            elif "Log" in name:
                dataset = "LogImagenet"
            else:
                dataset = "Imagenet"

            if "Log" in name:
                balance = "Log"
            elif "Unbalanced" in name:
                balance = "Linear"
            else:
                balance = "Balanced"

            if "_1000" in name:
                epochs = 1000
            elif "_800" in name or name == "PIRL":
                epochs = 800
            elif "_100" in name:
                epochs = 100
            elif "_50" in name:
                epochs = 50
            else:
                epochs = 200

            if "Half" in name:
                dataset_size = 500000
            elif "Unbalanced" in name:
                dataset_size = 500000
            elif "Quarter" in name:
                dataset_size = 250000
            elif "Log" in name:
                dataset_size = 250000
            else:
                dataset_size = 1000000

            row = {
                "Encoder": name,
                "Method": method,
                "Dataset": dataset,
                "Balance": balance,
                "Epochs": epochs,
                "Updates": epochs * dataset_size,
                "DatasetSize": dataset_size
            }
            row.update(results)
            writer.writerow(row)







### BIG TABLE
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# sns.set_theme()
# colors = sns.color_palette()
# palette = {method: colors[i] for i, method in enumerate(set(data["Method"]))}
# for task in ALL_TASKS:
#     plt.figure(figsize=(20, 10))
#     data = pandas.read_csv("results.csv")
#     results = data.sort_values(task, ascending=False).reset_index()
#     g = sns.barplot(x=task, y="Encoder", hue="Method", data=results, dodge=False, palette=palette)
#     sign = 1.0 if results[task][1] >= 0 else -1.0
#     for _, data in results.iterrows():
#         g.text(data[task] - (sign * 0.08), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
#     plt.title("%s Test Results" % task)
#     plt.xlabel("Test Performance")
#     #plt.show()
#     plt.savefig("graphs/task_by_task/%s-test-results-subtracted.png" % task, dpi=100)
#     plt.clf()
#
# values = []
# structural_values = []
# embedding_values = []
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# for task in ALL_TASKS:
#     rand = data.loc['Random', task]
#     for encoder in data.index:
#         data.loc[encoder, task] = data.loc[encoder, task] - rand
#     sup = data.loc['Supervised', task]
#     for encoder in data.index:
#         data.loc[encoder, task] = data.loc[encoder, task] / sup
# data = data.drop("Random")
# data = data.drop("SimCLR_1000")
#



#### Generating Pearson and Spearman Correlations
# make_csv()
# data = pandas.read_csv("results.csv")
# tasks = ["Imagenet", "Imagenetv2", "CalTech-101", "Pets", "dtd", "CIFAR-100", "SUN397", "KineticsActionPrediction", "Eurosat",
#          "THORNumSteps", "CLEVERNumObjects", "nuScenesActionPrediction", "THORActionPrediction",
#          "TaskonomyDepth", "NYUWalkable", "NYUDepth", "THORDepth",
#          "PetsDetection", "EgoHands", "CityscapesSemanticSegmentation"]
# n = len(tasks)
# spearman = np.zeros((n,n))
# pearson = np.zeros((n,n))
# spearman_pval = np.zeros((n,n))
# pearson_pval = np.zeros((n,n))
# for i in range(n):
#     for j in range(n):
#         print(tasks[i], tasks[j])
#         values_i = data[tasks[i]]
#         values_j = data[tasks[j]]
#         s, sp = scipy.stats.spearmanr(values_i, values_j)
#         p, pp = scipy.stats.pearsonr(values_i, values_j)
#         spearman[i][j] = s
#         pearson[i][j] = p
#         spearman_pval[i][j] = sp
#         pearson_pval[i][j] = pp
# # tasks = [t.replace("ActionPrediction", "").replace("SemanticSegmentation", "") for t in tasks]
# tasks = [REAL_NAMES[t] for t in tasks]
# # fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# # fig.suptitle("Correlation of End Task Performance Between Tasks")
# # ax = sns.heatmap(spearman, annot=False, ax=axes[0])
# # ax.set_title("Spearman")
# # ax.set_xticks([])
# # ax.set_yticks([])
# # ax = sns.heatmap(pearson, annot=False, ax=axes[1])
# # ax.set_title("Pearson")
# # ax.set_xticks([])
# # ax.set_yticks([])
# # plt.savefig("graphs/task_correlations/both.png")
#
# title = "Spearman Correlation on Performance Between Tasks"
# plt.figure(figsize=(13, 13))
# plt.title(title)
# ax = sns.heatmap(spearman, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/task_correlations/spearman.pdf", bbox_inches='tight')
# plt.clf()
# title = "Pearson Correlation on Performance Between Tasks"
# plt.figure(figsize=(13, 13))
# plt.title(title)
# ax = sns.heatmap(pearson, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/task_correlations/pearson.pdf", bbox_inches='tight')
# plt.clf()

# #### Generating Pearson and Spearman Correlations on Encoders Trained for 200 epochs
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder", drop=False)
# data = data.filter(regex="(MoCov2*|SWAV*)", axis=0)
# n = len(ALL_TASKS)
# spearman = np.zeros((n,n))
# pearson = np.zeros((n,n))
# spearman_pval = np.zeros((n,n))
# pearson_pval = np.zeros((n,n))
# for i in range(n):
#     for j in range(n):
#         values_i = data[ALL_TASKS[i]]
#         values_i -= np.min(values_i)
#         values_i /= np.max(values_i)
#         values_j = data[ALL_TASKS[j]]
#         values_j -= np.min(values_j)
#         values_j /= np.max(values_j)
#         s, sp = scipy.stats.spearmanr(values_i, values_j)
#         p, pp = scipy.stats.pearsonr(values_i, values_j)
#         spearman[i][j] = s
#         pearson[i][j] = p
#         spearman_pval[i][j] = sp
#         pearson_pval[i][j] = pp
#
# plt.figure(figsize=(20, 20))
# title = "Normalized Spearman Correlation on Performance Between Tasks"
# plt.title(title)
# ax = sns.heatmap(spearman, annot=True)
# ax.set_yticklabels(ALL_TASKS, rotation=0)
# ax.set_xticklabels(ALL_TASKS, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
# plt.clf()
# title = "Normalized Spearman Correlation p-values on Performance Between Tasks"
# plt.title(title)
# ax = sns.heatmap(spearman_pval, annot=True)
# ax.set_yticklabels(ALL_TASKS, rotation=0)
# ax.set_xticklabels(ALL_TASKS, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
# plt.clf()
# title = "Normalized Pearson Correlation on Performance Between Tasks on Encoders Trained for 200 Epochs"
# plt.title(title)
# ax = sns.heatmap(pearson, annot=True)
# ax.set_yticklabels(ALL_TASKS, rotation=0)
# ax.set_xticklabels(ALL_TASKS, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
# plt.clf()
# title = "Normalized Pearson Correlation p-values on Performance Between Tasks on Encoders Trained for 200 Epochs"
# plt.title(title)
# ax = sns.heatmap(pearson_pval, annot=True)
# ax.set_yticklabels(ALL_TASKS, rotation=0)
# ax.set_xticklabels(ALL_TASKS, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
# plt.clf()

# data = pandas.read_csv("results.csv")
# sns.set_theme()
# colors = sns.color_palette()
# palette = {method: colors[i] for i, method in enumerate(set(data["Method"]))}
# for task in ["Embedding", "Pixel"]:
#     data = pandas.read_csv("results.csv")
#     normalized_scores = get_normalized_summed_scores(data)
#     data = pandas.DataFrame(normalized_scores)
#     plt.figure(figsize=(20, 10))
#     results = data.sort_values(task, ascending=False).reset_index()
#     g = sns.barplot(x=task, y="Encoder", hue="Method", data=results, dodge=False, palette=palette)
#     sign = 1.0 if results[task][0] > 0 else -1.0
#     for _, data in results.iterrows():
#         g.text(data[task] - (sign * 0.02), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
#     plt.title("%s Test Results" % task)
#     plt.xlabel("Test Performance")
#     plt.savefig("graphs/%s-groupped-test-results.png" % task, dpi=100)
#     plt.clf()


# ### Generate average moco performance vs average swav performance
# data = pandas.read_csv("results.csv")
# # normalized_scores = get_normalized_summed_scores(data)
# # data = pandas.DataFrame(normalized_scores)
# data = data.set_index("Encoder")
# values = []
# for task in ALL_TASKS:
#     mocos_vals = []
#     swav_vals = []
#     for encoder in ALL_EXPERIMENTS:
#         encoder = encoder.replace("Imagenet", "IN")
#         if "MoCov2" in encoder:
#             mocos_vals.append(data[task][encoder])
#         if "SWAV" in encoder:
#             swav_vals.append(data[task][encoder])
#     values.append({"Method": "MoCov2", "task": task, "score": np.mean(mocos_vals)})
#     values.append({"Method": "SWAV", "task": task, "score": np.mean(swav_vals)})
# values = pandas.DataFrame(values)
# sns.set_theme()
# # normalized_scores = get_normalized_summed_scores(data)
# # data = pandas.DataFrame(normalized_scores)
# plt.figure(figsize=(20, 10))
# # results = data.sort_values(task, ascending=False).reset_index()
# g = sns.barplot(x="score", y="task", hue="Method", data=values)
# # sign = 1.0 if results[task][0] > 0 else -1.0
# # for _, data in results.iterrows():
# #     g.text(data[task] - (sign * 0.02), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
# plt.title("MoCo vs SWAV Test Results" )
# plt.xlabel("Test Performance")
# plt.show()
# # plt.savefig("graphs/%s-groupped-test-results.png" % task, dpi=100)
# plt.clf()

####### Make IN Subset Comparison Bar Charts
# values = []
# for task in ALL_TASKS:
#     experiments = HALF_IMAGENE_EXPERIMENTS + UNBALANCED_IMAGENET_EXPERIMENTS + IMAGENET_100_EPOCH_EXPERIMENTS
#     # experiments = QUARTER_IMAGENET_EXPERIMENTS + LOG_IMAGENET_EXPERIMENTS + IMAGENET_50_EPOCH_EXPERIMENTS + \
#     #               HALF_IMAGENE_100_EXPERIMENTS + UNBALANCED_IMAGENET_100_EXPERIMENTS
#     if task in REVERSED_SUCCESS_TASKS:
#         continue
#     data = get_best_result(experiments, task, include_names=True, c=(-1.0 if task in REVERSED_SUCCESS_TASKS else 1.0))
#     for encoder, result in data:
#         encoder = encoder.replace("Imagenet", "IN")
#         if "Half" in encoder:
#             values.append({"Encoder": encoder, "Dataset": "Half", "task": task, "score": result})
#         if "Quarter" in encoder:
#             values.append({"Encoder": encoder, "Dataset": "Quarter", "task": task, "score": result})
#         if "Unbalanced" in encoder:
#             values.append({"Encoder": encoder, "Dataset": "Unbalanced", "task": task, "score": result})
#         if "Log" in encoder:
#             values.append({"Encoder": encoder, "Dataset": "Log", "task": task, "score": result})
#         if "_100" in encoder:
#             values.append({"Encoder": encoder, "Dataset": "_100", "task": task, "score": result})
#         if "_50" in encoder:
#             values.append({"Encoder": encoder, "Dataset": "_50", "task": task, "score": result})
#
#         # values.append({"Dataset": "Half", "task": task, "score": np.mean(half_vals)})
#         # values.append({"Dataset": "Quarter", "task": task, "score": np.mean(quarter_vals)})
#         # values.append({"Dataset": "Unbalanced", "task": task, "score": np.mean(unbalanced_vals)})
#         # values.append({"Dataset": "Log", "task": task, "score": np.mean(log_vals)})
# values = pandas.DataFrame(values)
# sns.set_theme()
# # normalized_scores = get_normalized_summed_scores(data)
# # data = pandas.DataFrame(normalized_scores)
# plt.figure(figsize=(20, 10))
# # results = data.sort_values(task, ascending=False).reset_index()
# g = sns.barplot(x="score", y="task", hue="Encoder", data=values, dodge=True)
# # sign = 1.0 if results[task][0] > 0 else -1.0
# # for _, data in results.iterrows():
# #     g.text(data[task] - (sign * 0.02), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
# plt.title("Test Results of Encoders Trained on QuarterIN, LogIN, UnbalancedIN_100 and HalfIN_100 and FullIN_50")
# plt.xlabel("Test Performance")
# plt.show()
# # plt.savefig("graphs/%s-groupped-test-results.png" % task, dpi=100)
# plt.clf()

#### Generating Pearson and Spearman Correlations for IN Tasks
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder", drop=False)
#data = data.loc[ALL_EXPERIMENTS]
# tasks = ["Imagenet", "CalTech-101", "Pets", "CIFAR-100", "Pets-Detection", "dtd", "SUN397", "CLEVERNumObjects",
#           "NYUDepth", "NYUWalkable", "Eurosat", "THORDepth"]
# n = len(tasks)
# spearman = np.zeros((n,n))
# pearson = np.zeros((n,n))
# spearman_pval = np.zeros((n,n))
# pearson_pval = np.zeros((n,n))
# for i in range(n):
#     for j in range(n):
#         values_i = data[tasks[i]]
#         values_j = data[tasks[j]]
#         s, sp = scipy.stats.spearmanr(values_i, values_j)
#         p, pp = scipy.stats.pearsonr(values_i, values_j)
#         spearman[i][j] = s
#         pearson[i][j] = p
#         spearman_pval[i][j] = sp
#         pearson_pval[i][j] = pp
#
# plt.figure(figsize=(20, 20))
# title = "Spearman Correlation on Performance Between Tasks with non IN Encoders"
# plt.title(title)
# ax = sns.heatmap(spearman, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+"-1.png")
# plt.clf()
# title = "Spearman Correlation p-values on Performance Between Tasks with non IN Encoders"
# plt.title(title)
# ax = sns.heatmap(spearman_pval, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
# plt.clf()
# title = "Pearson Correlation on Performance Between Tasks  with non IN Encoders"
# plt.title(title)
# ax = sns.heatmap(pearson, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
# plt.clf()
# title = "Pearson Correlation p-values on Performance Between Tasks with non IN Encoders"
# plt.title(title)
# ax = sns.heatmap(pearson_pval, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
# plt.clf()

####### Plot the embedding end task results using the adam and sgd optimizers
# palette = {"sgd": "#38D9D9", "adam": "#DFEBEB"}
# for task in EMBEDDING_TASKS:
#     title="%s adam vs sgd" % task
#     res = get_all_results(ALL_EXPERIMENTS, task)
#     all_results = []
#     plt.figure(figsize=(20, 10))
#     plt.title(title)
#     for encoder, encoder_results in res.items():
#         if len(encoder_results) > 1:
#             best_run_with_optimizer = {"sgd": [], "adam":[]}
#             for run in encoder_results:
#                 best_run_with_optimizer[run["optimizer"]].append({
#                     "encoder": encoder+"-"+run["optimizer"],
#                     "optimizer": run["optimizer"],
#                     "lr": run["lr"],
#                     "score": run["result"]
#                 })
#             for _, runs in best_run_with_optimizer.items():
#                 runs.sort(key=lambda x: x["score"])
#                 all_results.append(runs[-1])
#
#     data = pandas.DataFrame(all_results)
#     data = data.sort_values("score")
#     g = sns.barplot(x="score", y="encoder", hue="optimizer", data=data, dodge=False, palette=palette)
#     plt.savefig("graphs/"+title.replace(" ", "_")+".png")
#     plt.clf()

#### Plot the number of times that each encoder ranks as first
# all_scores = {}
# for task in EMBEDDING_TASKS:
#     res = get_all_results(MOCOV2_200_EXPERIMENT+SWAV_200_EXPERIMENTS+["Supervised"], task)
#     all_results_for_task = []
#     for encoder, encoder_results in res.items():
#         if len(encoder_results) > 1:
#             best_run_with_optimizer = {"sgd": [], "adam": []}
#             for run in encoder_results:
#                 best_run_with_optimizer[run["optimizer"]].append({
#                     "encoder": encoder,
#                     "optimizer": run["optimizer"],
#                     "lr": run["lr"],
#                     "score": run["result"]
#                 })
#             best_run_with_optimizer["all"] = best_run_with_optimizer["sgd"] + best_run_with_optimizer["adam"]
#             for _, runs in best_run_with_optimizer.items():
#                 runs.sort(key=lambda x: x["score"])
#                 all_results_for_task.append(runs[-1])
#     all_results_for_task.sort(key=lambda x: x["score"], reverse=True)
#     sgd_results_for_task = [r for r in all_results_for_task if r["optimizer"] == "sgd"]
#     adam_results_for_task = [r for r in all_results_for_task if r["optimizer"] == "adam"]
#     all_scores[task] = {"sgd": sgd_results_for_task, "adam": adam_results_for_task, "all": all_results_for_task}
#
# num_first_place_table = {exp: 0 for exp in ALL_EXPERIMENTS}
# sgd_num_first_place_table = {exp: 0 for exp in ALL_EXPERIMENTS}
# adam_num_first_place_table = {exp: 0 for exp in ALL_EXPERIMENTS}
# for task, scores in all_scores.items():
#     num_first_place_table[scores["all"][0]["encoder"]] += 1
#     sgd_num_first_place_table[scores["sgd"][0]["encoder"]] += 1
#     adam_num_first_place_table[scores["adam"][0]["encoder"]] += 1
#
# num_first_list = []
# for enc, count in num_first_place_table.items():
#     if count > 0:
#         num_first_list.append({"encoder": enc, "count": count, "optimizer": "any"})
# for enc, count in sgd_num_first_place_table.items():
#     if count > 0:
#         num_first_list.append({"encoder": enc, "count": count, "optimizer": "sgd"})
# for enc, count in adam_num_first_place_table.items():
#     if count > 0:
#         num_first_list.append({"encoder": enc, "count": count, "optimizer": "adam"})
#
# title = "Number of first place rankings per encoder with any optimizer vs sgd for 200 epoch encoders"
# data = pandas.DataFrame(num_first_list)
# data = data.sort_values("count")
# plt.figure(figsize=(20, 10))
# plt.title(title)
# g = sns.barplot(y="count", x="encoder", hue="optimizer", data=data, dodge=True)
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
# plt.show()
# plt.clf()






# # ##### Plot Just SWAV 200 Data on different datasets
# data = pandas.read_csv("results.csv")
# swav_data = data[data["Method"] == "SWAV"]
# swav_200_data = swav_data[swav_data["Epochs"] == 200]
# swav_200_full_data = swav_200_data[swav_200_data["Updates"] == int(1e6 * 200)]
#
# plt.figure(figsize=(20, 10))
# sns.set_theme()
# colors = sns.color_palette()
# palette = {method: colors[i] for i, method in enumerate(set(data["Dataset"]))}
#
# for task in ALL_TASKS:
#     task_data = swav_200_full_data.sort_values(task, ascending=False).reset_index()
#     g = sns.barplot(x=task, y="Encoder", data=task_data, dodge=False, hue="Dataset", palette=palette)
#     sign = 1.0 if task_data[task][1] >= 0 else -1.0
#     for _, data in task_data.iterrows():
#         g.text(data[task] - (sign * 0.04), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
#     plt.title("%s Test Results" % task)
#     plt.xlabel("Test Performance")
#     #plt.show()
#     plt.savefig("graphs/swav_200_different_datasets/%s.png" % task, dpi=100)
#     plt.clf()
#
# ee = swav_200_full_data.set_index("Encoder")
# per_task_swav_200_full = []
# for task in ALL_TASKS:
#     for encoder in swav_200_full_data["Encoder"]:
#         if encoder == "SWAVKinetics" and task == "THORDepth":
#             continue
#         per_task_swav_200_full.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "NormalizedScore": ee.loc[encoder, task+"-normalized"],
#         })
# plt.title("SWAV 200 Modles, 1.3M Image Datasets Performance")
# df = pandas.DataFrame(per_task_swav_200_full)
# sns.swarmplot(x="Task", y="NormalizedScore", hue="Dataset", data=df, size=10)
# plt.xticks(rotation=30, rotation_mode="anchor")
# plt.savefig("graphs/swav_200_different_datasets/all.png", dpi=100)
# plt.clf()
#
#
# #### Plot Just MoCov2 200 Data on different datasets
# data = pandas.read_csv("results.csv")
# moco_data = data[data["Method"] == "MoCo"]
# moco_200_data = moco_data[moco_data["Epochs"] == 200]
# moco_200_full_data = moco_200_data[moco_200_data["Updates"] == int(1e6 * 200)]
#
# plt.figure(figsize=(20, 10))
# sns.set_theme()
# colors = sns.color_palette()
# # palette = {method: colors[i] for i, method in enumerate(set(data["Dataset"]))}
#
# for task in ALL_TASKS:
#     task_data = moco_200_full_data.sort_values(task, ascending=False).reset_index()
#     g = sns.barplot(x=task, y="Encoder", data=task_data, dodge=False, hue="Dataset", palette=palette)
#     sign = 1.0 if task_data[task][1] >= 0 else -1.0
#     for _, data in task_data.iterrows():
#         g.text(data[task] - (sign * 0.04), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
#     plt.title("%s Test Results" % task)
#     plt.xlabel("Test Performance")
#
#     plt.savefig("graphs/mocov2_200_different_datasets/%s.png" % task, dpi=100)
#     plt.clf()
#
# ee = moco_200_full_data.set_index("Encoder")
# per_task_moco_200_full = []
# for task in ALL_TASKS:
#     for encoder in moco_200_full_data["Encoder"]:
#         if encoder == "SWAVKinetics" and task == "THORDepth":
#             continue
#         per_task_moco_200_full.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "NormalizedScore": ee.loc[encoder, task+"-normalized"],
#         })
# plt.title("MoCov2 200 Modles, 1.3M Image Datasets Performance")
# df = pandas.DataFrame(per_task_moco_200_full)
# sns.swarmplot(x="Task", y="NormalizedScore", hue="Dataset", data=df, size=10, palette=palette)
# plt.xticks(rotation=30, rotation_mode="anchor")
# plt.savefig("graphs/mocov2_200_different_datasets/all.png", dpi=100)
# plt.clf()


##### Plot MoCov2 200 and SWAV Data on different datasets
# data = pandas.read_csv("results.csv")
# ms_data = data[(data["Method"] == "MoCo") | (data["Method"] == "SWAV")]
# ms_200_data = ms_data[ms_data["Epochs"] == 200]
# ms_200_full_data = ms_data[ms_data["Updates"] == int(1e6 * 200)]
#
# sns.set_theme()
# colors = sns.color_palette()
# # palette = {method: colors[i] for i, method in enumerate(set(data["Dataset"]))}
# plt.figure(figsize=(20, 10))
#
# ee = ms_200_full_data.set_index("Encoder")
# per_task_ms_200_full = []
# for task in ALL_TASKS:
#     for encoder in ms_200_full_data["Encoder"]:
#         if encoder == "SWAVKinetics" and task == "THORDepth":
#             continue
#         per_task_ms_200_full.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "NormalizedScore": ee.loc[encoder, task+"-normalized"],
#         })
# plt.title("SWAV and MoCov2 200 Modles, 1.3M Image Datasets Performance")
# df = pandas.DataFrame(per_task_ms_200_full)
# sns.scatterplot(y="Task", x="NormalizedScore", hue="Dataset", data=df, style="Method", s=100)
#
# plt.savefig("graphs/mocov2_and_swav_200_different_datasets/all.png", dpi=100)
# plt.clf()


#$##### Figure 6. Make Per Task Vategory Violin Plot of Scores for MoCo and SWAV 200
# data = pandas.read_csv("results.csv")
# ms_data = data[(data["Method"] == "MoCo") | (data["Method"] == "SWAV") | (data["Method"] == "PIRL")]
# ms_200_data = ms_data[(ms_data["Epochs"] == 200) | (ms_data["Epochs"] == 800)]
# ms_200_full_data = ms_200_data[ms_200_data["Updates"] == int(1e6 * 200)]
#
#
# sns.set_theme()
# colors = sns.color_palette()
# # palette = {method: colors[i] for i, method in enumerate(set(data["Dataset"]))}
# plt.figure(figsize=(10, 10))
#
# ee = ms_200_full_data.set_index("Encoder")
# per_task_ms_200_full = []
# for task in PIXELWISE_SEMANTIC_TASKS + EMBEDDING_SEMANTIC_TASKS + PIXELWISE_STRUCTURAL_TASKS + EMBEDDING_STRUCTURAL_TASKS:
#     for encoder in ms_200_full_data["Encoder"]:
#         if encoder == "SWAVKinetics" and task == "THORDepth":
#             continue
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Image-Level Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Image-Level Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise Structural"
#         else:
#             ttype = "Other"
#
#         per_task_ms_200_full.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "Normalized End Task Performance": ee.loc[encoder, task+"-normalized"],
#             "Task Type": ttype
#         })
#
# sns.set_theme(style="whitegrid", font_scale=1.5)
# pal = sns.color_palette("GnBu")
# df = pandas.DataFrame(per_task_ms_200_full)
# order = ["Image-Level Semantic", "Pixelwise Semantic", "Image-Level Structural", "Pixelwise Structural"]
# ax = sns.violinplot(x="Task Type", y="Normalized End Task Performance", hue="Task Type", data=df, dodge=False, order=order,
#                     palette={"Image-Level Semantic": pal[0], "Pixelwise Semantic": pal[1],
#                              "Image-Level Structural": pal[4], "Pixelwise Structural": pal[5]}
#                     )
# ax = sns.lineplot(x=[-1,4], y=[0,0], hue=["Supervised ImageNet", "Supervised ImageNet"], linewidth=3,
#                   style=["Supervised ImageNet", "Supervised ImageNet"],
#                   dashes=[(2,2)],
#                   palette={"Supervised ImageNet": 'red'})
# ax.set_xlim(-1,4)
# ax.set(xticklabels=[])
# # ax.axhline(0.0)
# ax.legend(loc='upper left')
#
#
# # plt.show()
# plt.savefig("graphs/mocov2_and_swav_200_different_task_types/violin.pdf", bbox_inches='tight')
# # plt.clf()


##### Make Per Task Type Violin Plot of Scores for MoCo and SWAV 200
# make_csv()
# data = pandas.read_csv("results.csv")
# ms_data = data[(data["Method"] == "MoCo") | (data["Method"] == "SWAV")]
# ms_data = ms_data[ms_data["Epochs"] == 200]
# ms_data = ms_data[ms_data["Updates"] == int(1e6 * 200)]
#
# sns.set()
# sns.set_style("whitegrid", {'axes.grid' : False})
# colors = sns.color_palette()
# # palette = {method: colors[i] for i, method in enumerate(set(data["Dataset"]))}
#
# ee = ms_data.set_index("Encoder")
# per_task_ms_200_full = []
# for task in PIXELWISE_SEMANTIC_TASKS + EMBEDDING_SEMANTIC_TASKS + PIXELWISE_STRUCTURAL_TASKS + EMBEDDING_STRUCTURAL_TASKS:
#     for encoder in ms_data["Encoder"]:
#         if encoder == "SWAVKinetics" and task == "THORDepth":
#             continue
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "PixelWise"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "PixelWise"
#         else:
#             ttype = "Other"
#
#         # if ttype != "Embedding":
#         #     continue
#
#         per_task_ms_200_full.append({
#             "Encoder": encoder,
#             "Task": task.replace("SemanticSegmentation", ""),
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "INScore": ee.loc[encoder, "Imagenet"],
#             "Score": ee.loc[encoder, task],
#             "NormalizedScore": ee.loc[encoder, task+"-normalized"],
#             "TaskType": ttype
#         })
# df = pandas.DataFrame(per_task_ms_200_full)
# # ax = sns.displot(x="Score", col="Task", hue="Method", kind="kde", data=df, height=2, aspect=2, col_wrap=3)
# # ax.set(xlim=(0, 0.5))
# # plt.subplots_adjust(hspace = 0.2)
# # all_tasks = EMBEDDING_SEMANTIC_TASKS + EMBEDDING_STRUCTURAL_TASKS
# # all_tasks = ["TaskonomyDepth","NYUDepth", "NYUWalkable", "THORDepth", "KITTI"]
# # all_tasks = ["Cityscapes","EgoHands", "PetsDetection"]
# # all_tasks = ["CLEVERNumObjects", "dtd", "SUN397", "CIFAR-100", "nuScenesActionPrediction", "THORNumSteps","Eurosat", "Imagenet", "Pets", "CalTech-101", "THORActionPrediction"]
# sns.scatterplot(x="Score", y="Task", hue="Method", style="TaskType", data=df)
# # fig, axes = plt.subplots(len(all_tasks), 1)
# # fig.suptitle("Top 1 Accuracy")
# # for i, task in enumerate(all_tasks):
# #     pts = [p for p in per_task_ms_200_full if p["Task"] == task]
# #     sns.scatterplot(x="NormalizedScore", hue="Method", ax=axes[i], data=pandas.DataFrame(pts), fill=True, common_norm=False, legend=False, hue_order=["MoCo", "SWAV"])
# #     axes[i].axis('off')
# plt.show()
# exit()




# ####### Figure 5. Make SWAV vs MoCo Performance Plots
# make_csv()
# data = pandas.read_csv("results.csv")
# ms_data = data[(data["Method"] == "MoCo") | (data["Method"] == "SWAV")]
# ms_data = ms_data[ms_data["Epochs"] == 200]
# ms_data = ms_data[ms_data["Updates"] == int(1e6 * 200)]
# ee = ms_data.set_index("Encoder")
# diffs = []
# for task in PIXELWISE_SEMANTIC_TASKS + EMBEDDING_SEMANTIC_TASKS + PIXELWISE_STRUCTURAL_TASKS + EMBEDDING_STRUCTURAL_TASKS:
#     moco = []
#     swav = []
#     if task == "Imagenetv2":
#         continue
#     for encoder in ms_data["Encoder"]:
#         if math.isnan(ee.loc[encoder, task]):
#             continue
#         if encoder == "SWAVKinetics" and task == "THORDepth":
#             continue
#         if ee.loc[encoder, "Method"] == "MoCo":
#             moco.append(ee.loc[encoder, task])
#         elif ee.loc[encoder, "Method"] == "SWAV":
#             swav.append(ee.loc[encoder, task])
#
#     if task in EMBEDDING_SEMANTIC_TASKS:
#         ttype = "Image-level"
#     elif task in EMBEDDING_STRUCTURAL_TASKS:
#         ttype = "Image-level"
#     elif task in PIXELWISE_SEMANTIC_TASKS:
#         ttype = "PixelWise"
#     elif task in PIXELWISE_STRUCTURAL_TASKS:
#         ttype = "PixelWise"
#     else:
#         ttype = "Other"
#
#     if task in ["Cityscapes", "PetsDetection", "EgoHands", "NYUWalkable"]:
#         metric = "mIOU"
#     elif task in ["TaskonomyDepth", "NYUDepth", "THORDepth", "KITTI"]:
#         metric = "L1 Error"
#     else:
#         metric = "Top 1 Accuracy"
#
#     diffs.append({
#         "Task": REAL_NAMES[task],
#         "Method": "MoCo",
#         "AverageScore": (np.mean(moco) - np.mean(swav)) / np.abs(np.mean(swav)) * 100,
#         "TaskType": ttype,
#         "Metric": metric
#     })
#
#     # diffs.append({
#     #     "Task": task.replace("SemanticSegmentation", ""),
#     #     "Method": "SWAV",
#     #     "AverageScore": np.mean(swav),
#     #     "TaskType": ttype,
#     #     "Metric": metric
#     # })
#
# sns.set()
# sns.set_theme(style="whitegrid", font_scale=1.8, palette="pastel")
# plt.figure(figsize=(10, 10))
# # plt.title("Average Performance of MoCov2 and SWAV Encoders on End Tasks")
# order = ["CityscapesSemanticSegmentation", "TaskonomyDepth", "PetsDetection",
#          "NYUWalkable", "NYUDepth", "EgoHands", "KITTI", "THORDepth",
#          'KineticsActionPrediction', "CLEVERNumObjects", "dtd", "Imagenet", "SUN397", "CIFAR-100", "Pets",
#          "nuScenesActionPrediction", "THORNumSteps", "CalTech-101", "Eurosat", "THORActionPrediction"]
# order = [REAL_NAMES[o] for o in order]
# diffs.sort(key=lambda x: order.index(x["Task"]))
# df = pandas.DataFrame(diffs)
# ax = sns.barplot(x="AverageScore", y="Task", hue="TaskType", dodge=False, data=df)
# ax.arrow(-9, 18, -1.5, 0, color="black", width=0.1)
# ax.text(-8.8, 18.3, "SwAV Better")
# ax.arrow(8, 1, 1.5, 0, color="black", width=0.1)
# ax.text(1.5, 1.3, "MoCov2 Better")
# plt.ylabel("End Task")
# plt.xlabel("Relative percentage difference between average MoCov2 and SwAV Score")
# # plt.show()
# plt.savefig("graphs/mocov2_and_swav_200_different_task_types/diff.pdf", bbox_inches='tight')
# # plt.clf()


##### Make Per Task Vategory Violin Plot of Scores for MoCo 200
# data = pandas.read_csv("results.csv")
# ms_data = data[(data["Method"] == "MoCo")]
# ms_200_data = ms_data[ms_data["Epochs"] == 200]
# ms_200_full_data = ms_200_data[ms_200_data["Updates"] == int(1e6 * 200)]
#
# sns.set_theme()
# sns.set_theme(style="whitegrid", font_scale=1.8, palette='pastel')
# plt.figure(figsize=(20, 10))
#
# ee = ms_200_full_data.set_index("Encoder")
# per_task_ms_200_full = []
# for task in PIXELWISE_SEMANTIC_TASKS + EMBEDDING_SEMANTIC_TASKS + PIXELWISE_STRUCTURAL_TASKS + EMBEDDING_STRUCTURAL_TASKS:
#     for encoder in ms_200_full_data["Encoder"]:
#         if encoder == "SWAVKinetics" and task == "THORDepth":
#             continue
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         per_task_ms_200_full.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "NormalizedScore": ee.loc[encoder, task+"-normalized"],
#             "TaskType": ttype
#         })
#
# plt.title("MoCov2 200 Models, 1.3M Image Datasets Per Category Performance")
# df = pandas.DataFrame(per_task_ms_200_full)
# sns.boxplot(x="Task", y="NormalizedScore", hue="TaskType", data=df)
# plt.show()
# plt.savefig("graphs/mocov2_200_different_datasets/violin.png", dpi=100)
# plt.clf()

##### Make Per Task Vategory Violin Plot of Scores for SWAV 200
# data = pandas.read_csv("results.csv")
# ms_data = data[(data["Method"] == "SWAV")]
# ms_200_data = ms_data[ms_data["Epochs"] == 200]
# ms_200_full_data = ms_data[ms_data["Updates"] == int(1e6 * 200)]
#
# sns.set_theme()
# colors = sns.color_palette()
# plt.figure(figsize=(20, 10))
#
# ee = ms_200_full_data.set_index("Encoder")
# per_task_ms_200_full = []
# for task in PIXELWISE_SEMANTIC_TASKS + EMBEDDING_SEMANTIC_TASKS + PIXELWISE_STRUCTURAL_TASKS + EMBEDDING_STRUCTURAL_TASKS:
#     for encoder in ms_200_full_data["Encoder"]:
#         if encoder == "SWAVKinetics" and task == "THORDepth":
#             continue
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         per_task_ms_200_full.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "NormalizedScore": ee.loc[encoder, task+"-normalized"],
#             "TaskType": ttype
#         })
#
# plt.title("SWAV 200 Models, 1.3M Image Datasets Per Category Performance")
# df = pandas.DataFrame(per_task_ms_200_full)
# sns.violinplot(x="Task", y="NormalizedScore", hue="TaskType", data=df)
# plt.savefig("graphs/swav_200_different_datasets/violin.png", dpi=100)
# plt.clf()


# data = pandas.read_csv("results.csv")
# sns.set_theme()
# colors = sns.color_palette()
# for dataset in data.Dataset.unique():
#     d = data[(data["Dataset"] == dataset) & (data["Epochs"] == 200) & (data["Updates"] == int(200*1e6))]
#     d = d[(d["Encoder"] != "Supervised") & (d["Encoder"] != "Random") & (d["Encoder"] != "PIRL")]
#     plt.figure(figsize=(20, 10))
#
#     ee = d.set_index("Encoder")
#     vals = []
#     for task in [t for t in ALL_TASKS if t not in REVERSED_SUCCESS_TASKS]:
#         for encoder in d["Encoder"]:
#             if encoder == "SWAVKinetics" and task == "THORDepth":
#                 continue
#
#             if task in EMBEDDING_SEMANTIC_TASKS:
#                 ttype = "Embedding-Semantic"
#             elif task in EMBEDDING_STRUCTURAL_TASKS:
#                 ttype = "Embedding-Structural"
#             elif task in PIXELWISE_SEMANTIC_TASKS:
#                 ttype = "Pixelwise-Semantic"
#             elif task in PIXELWISE_STRUCTURAL_TASKS:
#                 ttype = "Pixelwise-Structural"
#             else:
#                 ttype = "Other"
#
#             vals.append({
#                 "Encoder": encoder,
#                 "Task": task,
#                 "Method": ee.loc[encoder, "Method"],
#                 "Dataset": ee.loc[encoder, "Dataset"],
#                 "Score": ee.loc[encoder, task],
#                 "SupervisedScore": data.set_index("Encoder").loc["Supervised", task],
#                 "TaskType": ttype
#             })
#
#     plt.title("Models trained on %s for 200 epochs Performance vs. Supervised" % dataset)
#     df = pandas.DataFrame(vals)
#     g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Method", style="TaskType", data=df, s=100)
#     sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["IN Results", "IN Results"], palette={"IN Results": "Red"})
#     plt.xlabel("Supervised Performance")
#     plt.ylabel("Models Trained on %s Dataset Model Performance" % dataset)
#     plt.savefig("graphs/different_datasets/%s_vs_Supervised.png" % dataset, dpi=100)
#     plt.clf()
#
#
# d = data[(data["Epochs"] == 200) & (data["Updates"] == int(200*1e6))]
# d = d[(d["Encoder"] != "Supervised") & (d["Encoder"] != "Random") & (d["Encoder"] != "PIRL")]
# plt.figure(figsize=(20, 10))
# ee = d.set_index("Encoder")
# vals = []
# for task in [t for t in ALL_TASKS if t not in REVERSED_SUCCESS_TASKS]:
#     for encoder in d["Encoder"]:
#         if encoder == "SWAVKinetics" and task == "THORDepth":
#             continue
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         vals.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "Score": ee.loc[encoder, task],
#             "SupervisedScore": data.set_index("Encoder").loc["Supervised", task],
#             "TaskType": ttype
#         })
#
# plt.title("Models trained on All Datasets for 200 epochs Performance vs. Supervised")
# df = pandas.DataFrame(vals)
# g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Dataset", style="Method", data=df, s=100)
# sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["IN Results", "IN Results"], palette={"IN Results": "Red"})
# plt.xlabel("Supervised Performance")
# plt.ylabel("Self Supervised Encoder Model Performance")
# plt.savefig("graphs/different_datasets/All_vs_Supervised.png", dpi=100)
# plt.clf()
#
# data = pandas.read_csv("results.csv")
# sns.set_theme()
# colors = sns.color_palette()
# for dataset in data.Dataset.unique():
#     d = data[(data["Dataset"] == dataset) & (data["Epochs"] == 200) & (data["Updates"] == int(200*1e6))]
#     d = d[(d["Encoder"] != "Supervised") & (d["Encoder"] != "Random") & (d["Encoder"] != "PIRL")]
#     plt.figure(figsize=(20, 10))
#
#     ee = d.set_index("Encoder")
#     vals = []
#     for task in [t for t in ALL_TASKS if t not in REVERSED_SUCCESS_TASKS]:
#         for encoder in d["Encoder"]:
#             if encoder == "SWAVKinetics" and task == "THORDepth":
#                 continue
#
#             if task in EMBEDDING_SEMANTIC_TASKS:
#                 ttype = "Embedding-Semantic"
#             elif task in EMBEDDING_STRUCTURAL_TASKS:
#                 ttype = "Embedding-Structural"
#             elif task in PIXELWISE_SEMANTIC_TASKS:
#                 ttype = "Pixelwise-Semantic"
#             elif task in PIXELWISE_STRUCTURAL_TASKS:
#                 ttype = "Pixelwise-Structural"
#             else:
#                 ttype = "Other"
#
#             vals.append({
#                 "Encoder": encoder,
#                 "Task": task,
#                 "Method": ee.loc[encoder, "Method"],
#                 "Dataset": ee.loc[encoder, "Dataset"],
#                 "Score": ee.loc[encoder, task],
#                 "SupervisedScore": data.set_index("Encoder").loc["Supervised", task],
#                 "TaskType": ttype
#             })
#
#     plt.title("Models trained on %s for 200 epochs Performance vs. Supervised" % dataset)
#     df = pandas.DataFrame(vals)
#     g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Method", style="TaskType", data=df, s=100)
#     sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["IN Results", "IN Results"], palette={"IN Results": "Red"})
#     plt.xlabel("Supervised Performance")
#     plt.ylabel("Models Trained on %s Dataset Model Performance" % dataset)
#     plt.savefig("graphs/Imagenet_Subsets/%s_vs_Supervised.png" % dataset, dpi=100)
#     plt.clf()
#
#
# d = data[(data["Updates"] == int(100*1e6)) & (data["Method"] == "MoCo")]
# plt.figure(figsize=(20, 10))
# ee = d.set_index("Encoder")
# vals = []
# for task in ALL_TASKS:
#     for encoder in d["Encoder"]:
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         vals.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "Score": ee.loc[encoder, task],
#             "SupervisedScore": data.set_index("Encoder").loc["MoCov2_100", task],
#             "TaskType": ttype
#         })
#
# plt.title("Half Imagenet and Equivalent MoCo Enocders Performance")
# df = pandas.DataFrame(vals)
# g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Dataset", data=df, style="TaskType", s=100)
# sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["MoCov2_100", "MoCov2_100"], palette={"MoCov2_100": "Red"})
# plt.xlabel("MoCov2_100 Performance")
# plt.ylabel("Self Supervised Encoder Model Performance")
# plt.savefig("graphs/Imagenet_Subsets/MoCov2HalfEquivalent_vs_Supervised.png", dpi=100)
# plt.clf()
#
#
# d = data[(data["Updates"] == int(50*1e6)) & (data["Method"] == "MoCo")]
# plt.figure(figsize=(20, 10))
# ee = d.set_index("Encoder")
# vals = []
# for task in ALL_TASKS:
#     for encoder in d["Encoder"]:
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         vals.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "Score": ee.loc[encoder, task],
#             "SupervisedScore": data.set_index("Encoder").loc["MoCov2_50", task],
#             "TaskType": ttype
#         })
#
# plt.title("Quarter Imagenet and Equivalent MoCo Enocders Performance")
# df = pandas.DataFrame(vals)
# g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Dataset", data=df, style="TaskType", s=100)
# sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["MoCov2_50", "MoCov2_50"], palette={"MoCov2_50": "Red"})
# plt.xlabel("MoCov2_50")
# plt.ylabel("Self Supervised Encoder Model Performance")
# plt.savefig("graphs/Imagenet_Subsets/MoCov2QuarterEquivalent_vs_Supervised.png", dpi=100)
# plt.clf()
#
#
# d = data[(data["Updates"] == int(100*1e6)) & (data["Method"] == "MoCo")]
# plt.figure(figsize=(20, 10))
# ee = d.set_index("Encoder")
# vals = []
# for task in [t for t in ALL_TASKS if t not in REVERSED_SUCCESS_TASKS]:
#     for encoder in d["Encoder"]:
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         vals.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "Score": ee.loc[encoder, task],
#             "SupervisedScore": data.set_index("Encoder").loc["MoCov2_100", task],
#             "TaskType": ttype
#         })
#
# plt.title("Half Imagenet and Equivalent MoCo Enocders Performance")
# df = pandas.DataFrame(vals)
# g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Dataset", data=df, s=100)
# sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["MoCov2_100", "MoCov2_100"], palette={"MoCov2_100": "Red"})
# plt.xlabel("MoCov2_100 Performance")
# plt.ylabel("Self Supervised Encoder Model Performance")
# plt.savefig("graphs/Imagenet_Subsets/MoCov2HalfEquivalent_vs_Supervised.png", dpi=100)
# plt.clf()
#
#
# d = data[(data["Updates"] == int(100*1e6)) & (data["Method"] == "SWAV")]
# plt.figure(figsize=(20, 10))
# ee = d.set_index("Encoder")
# vals = []
# for task in [t for t in ALL_TASKS if t not in REVERSED_SUCCESS_TASKS]:
#     for encoder in d["Encoder"]:
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         vals.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "Score": ee.loc[encoder, task],
#             "SupervisedScore": data.set_index("Encoder").loc["SWAV_100", task],
#             "TaskType": ttype
#         })
#
# plt.title("Half Imagenet and Equivalent SWAV Enocders Performance")
# df = pandas.DataFrame(vals)
# g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Dataset", data=df, s=100)
# sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["SWAV_100", "SWAV_100"], palette={"SWAV_100": "Red"})
# plt.xlabel("SWAV_100 Performance")
# plt.ylabel("Self Supervised Encoder Model Performance")
# plt.savefig("graphs/Imagenet_Subsets/SWAVHalfEquivalent_vs_Supervised.png", dpi=100)
# plt.clf()
#
#
# d = data[(data["Updates"] == int(50*1e6)) & (data["Method"] == "SWAV")]
# plt.figure(figsize=(20, 10))
# ee = d.set_index("Encoder")
# vals = []
# for task in ALL_TASKS:
#     for encoder in d["Encoder"]:
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         vals.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "Score": ee.loc[encoder, task],
#             "SupervisedScore": data.set_index("Encoder").loc["SWAV_50", task],
#             "TaskType": ttype
#         })
#
# plt.title("Quarter Imagenet and Equivalent SWAV Enocders Performance")
# df = pandas.DataFrame(vals)
# g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Dataset", data=df, style="TaskType", s=100)
# sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["SWAV_50", "SWAV_50"], palette={"SWAV_50": "Red"})
# plt.xlabel("MoCov2_50")
# plt.ylabel("Self Supervised Encoder Model Performance")
# plt.savefig("graphs/Imagenet_Subsets/SWAVQuarterEquivalent_vs_Supervised.png", dpi=100)
# plt.clf()
#
# #
# data = pandas.read_csv("results.csv")
# d = data[(data["Encoder"] == "SWAV_800") | (data["Encoder"] == "SWAV_200") |
#          (data["Encoder"] == "SWAV_100") | (data["Encoder"] == "SWAV_50") |
#          # (data["Encoder"] == "SWAVHalfIN") | (data["Encoder"] == "SWAVUnbalancedIN") |
#          # (data["Encoder"] == "SWAVHalfIN_100") | (data["Encoder"] == "SWAVUnbalancedIN_100") |
#          (data["Encoder"] == "MoCov2_800") | (data["Encoder"] == "MoCov2_200") |
#          (data["Encoder"] == "MoCov2_100") | (data["Encoder"] == "MoCov2_50")
#          # (data["Encoder"] == "MoCov2HalfIN") | (data["Encoder"] == "MoCov2nbalancedIN") |
#          # (data["Encoder"] == "MoCov2HalfIN_100") | (data["Encoder"] == "MoCov2UnbalancedIN_100")
# ]
# plt.figure(figsize=(20, 10))
# ee = d.set_index("Encoder")
# vals = []
# for task in ALL_TASKS:
#     for encoder in d["Encoder"]:
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         vals.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "Score": ee.loc[encoder, task+"-normalized"],
#             "TaskType": ttype,
#             "Updates": ee.loc[encoder, "Updates"] / 50000000,
#         })
#
# plt.title("MoCo and SWAV Enocders with different amounts of training steps Performance")
# df = pandas.DataFrame(vals)
# g = sns.scatterplot(y="Task", x="Score", hue="Method", data=df, size="Updates", sizes={1:50, 2:100, 4:150, 8:200, 16:250})
# plt.xlabel("Normalized Score")
# plt.savefig("graphs/Imagenet_Subsets/INSubsetModels_Differnt_Number_of_Updates.png", dpi=100)
# plt.clf()

# data = pandas.read_csv("results.csv")
# d = data[
#          # (data["Encoder"] == "SWAV_800") | (data["Encoder"] == "SWAV_200") |
#          # (data["Encoder"] == "SWAV_100") | (data["Encoder"] == "SWAV_50") |
#          (data["Encoder"] == "SWAVHalfIN") | (data["Encoder"] == "SWAVUnbalancedIN") |
#          (data["Encoder"] == "SWAVHalfIN_100") | (data["Encoder"] == "SWAVUnbalancedIN_100") |
#          # (data["Encoder"] == "MoCov2_800") | (data["Encoder"] == "MoCov2_200") |
#          # (data["Encoder"] == "MoCov2_100") | (data["Encoder"] == "MoCov2_50")
#          (data["Encoder"] == "MoCov2HalfIN") | (data["Encoder"] == "MoCov2nbalancedIN") |
#          (data["Encoder"] == "MoCov2HalfIN_100") | (data["Encoder"] == "MoCov2UnbalancedIN_100")
# ]
# plt.figure(figsize=(20, 10))
# ee = d.set_index("Encoder")
# vals = []
# for task in ALL_TASKS:
#     for encoder in ["SWAVHalfIN"]:
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         vals.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "Score": ee.loc[encoder, task+"-normalized"],
#             "TaskType": ttype,
#             "Updates": ee.loc[encoder, "Updates"] / 50000000,
#         })
# for task in ALL_TASKS:
#     for encoder in d["Encoder"]:
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         vals.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "Score": ee.loc[encoder, task+"-normalized"],
#             "TaskType": ttype,
#             "Updates": ee.loc[encoder, "Updates"] / 50000000,
#         })
#
# plt.title("MoCo and SWAV Enocders with different amounts of training steps Performance")
# df = pandas.DataFrame(vals)
# g = sns.scatterplot(y="Task", x="Score", hue="Method", data=df, size="Updates", sizes={1:50, 2:100, 4:150, 8:200, 16:250})
# plt.show()
# plt.savefig("graphs/Imagenet_Subsets/PerformanceImprovementWithMoreTraining.png", dpi=100)
# plt.clf()
#
#
#
#
#
########## Plot The Performance of MoCo and SWAV Subset models based off the amount of training data
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# data = data.loc[["SWAV_200", "MoCov2_200", "SWAVHalfIN", "MoCov2HalfIN", "SWAVUnbalancedIN", "MoCov2UnbalancedIN",
#                  "SWAVQuarterIN", "MoCov2QuarterIN", "SWAVLogIN", "MoCov2LogIN"]]
# plt.figure(figsize=(20, 10))
# sns.set_theme()
# colors = sns.color_palette()
# palette = {method: colors[i] for i, method in enumerate(set(data["DatasetSize"]))}
#
# for task in ALL_TASKS:
#     data = pandas.read_csv("results.csv")
#     data = data.set_index("Encoder")
#     data = data.loc[["SWAV_200", "MoCov2_200", "SWAVHalfIN", "MoCov2HalfIN", "SWAVUnbalancedIN", "MoCov2UnbalancedIN",
#                      "SWAVQuarterIN", "MoCov2QuarterIN", "SWAVLogIN", "MoCov2LogIN"]]
#     task_data = data.reset_index()
#     g = sns.barplot(x=task, y="Encoder", data=task_data, dodge=False, hue="DatasetSize", palette=palette)
#     sign = 1.0 if task_data[task][1] >= 0 else -1.0
#     for _, data in task_data.iterrows():
#         g.text(data[task] - (sign * 0.04), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
#     plt.title("%s Test Results" % task)
#     plt.xlabel("Test Performance")
#     plt.savefig("graphs/Imagenet_Subsets/%s.png" % task, dpi=100)
#     plt.clf()
#
#
#
# ###### Make graph of performance difference between Imagenetv1 and Imagenet v2 test results
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# values = []
# plt.figure(figsize=(20, 10))
# sns.set_theme()
# for exp in ALL_EXPERIMENTS:
#     exp = exp.replace("Imagenet", "IN")
#     values.append({
#         "Encoder": exp,
#         "Score": -100 * (data.loc[exp]["Imagenet"] - data.loc[exp]["Imagenetv2"])
#     })
#     # values.append({"Encoder": exp, "Score": data[exp]["Imagenet"], "Task": "Imagenet v1 Test"})
#     # values.append({"Encoder": exp, "Score": data[exp]["Imagenetv2"], "Task": "Imagenet v2 Test"})
# data = pandas.DataFrame(values)
# data = data.set_index("Encoder")
# data = data.drop("PIRL")
# data = data.drop("SimCLR_1000")
# data = data.sort_values("Score", ascending=False).reset_index()
# plt.title("INv1 vs INv2 Test Set Performance Difference")
# ax = sns.barplot(y="Encoder", x="Score", data=data)
# plt.xlabel("Ansolute difference between INv1 and INv2 test performance")
# plt.savefig("graphs/inv1_vs_inv2/absolute_difference.png")
# plt.clf()
#
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# values = []
# plt.figure(figsize=(20, 10))
# sns.set_theme()
# for exp in ALL_EXPERIMENTS:
#     exp = exp.replace("Imagenet", "IN")
#     values.append({
#         "Encoder": exp,
#         "INv1": data.loc[exp]["Imagenet"],
#         "INv2": data.loc[exp]["Imagenetv2"]
#     })
# data = pandas.DataFrame(values)
# data = data.set_index("Encoder")
# data = data.drop("PIRL")
# data = data.drop("SimCLR_1000")
# data = data.drop("Random")
# # data = data.sort_values("Score", ascending=False).reset_index()
# plt.title("INv1 vs INv2 Test Set Performance Trend")
# ax = sns.scatterplot(y="INv2", x="INv1", data=data, label="Model accuracy")
# ax = sns.regplot(y="INv2", x="INv1", color="Red", label="Linear fit", scatter=False, data=data, ax=ax)
# ax = sns.lineplot(x=[0.28, 0.78], y=[0.28, 0.78], color="Black", label="Ideal reproducibility ")
# plt.xlabel("INv1 Test Set Performance")
# plt.ylabel("INv2 Test Set Performance")
# plt.savefig("graphs/inv1_vs_inv2/trend.png")
# plt.clf()

# data = pandas.read_csv("results.csv")
# # Select Quarter and Half Equivalent models
# data = data[(data["Updates"] == int(50*1e6))]
# tasks = ["Imagenet", "CalTech-101", "Pets", "PetsDetection", "dtd", "CIFAR-100", "SUN397", "Eurosat",
#          "CLEVERNumObjects", "THORNumSteps", "THORDepth", "NYUDepth", "NYUWalkable", "THORActionPrediction"]
# n = len(tasks)
# spearman = np.zeros((n,n))
# pearson = np.zeros((n,n))
# spearman_pval = np.zeros((n,n))
# pearson_pval = np.zeros((n,n))
# for i in range(n):
#     for j in range(n):
#         values_i = data[tasks[i]]
#         values_j = data[tasks[j]]
#         s, sp = scipy.stats.spearmanr(values_i, values_j)
#         p, pp = scipy.stats.pearsonr(values_i, values_j)
#         spearman[i][j] = s
#         pearson[i][j] = p
#         spearman_pval[i][j] = sp
#         pearson_pval[i][j] = pp
#
# plt.figure(figsize=(20, 20))
# title = "Spearman Correlation on Performance Between Tasks of Quarter IN Equivalent Models"
# plt.title(title)
# ax = sns.heatmap(spearman, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/Imagenet_Subsets/QuarterSpearman.png")
# plt.clf()
# title = "Spearman Correlation p-values on Performance Between Tasks of Quarter IN Equivalent Models"
# plt.title(title)
# ax = sns.heatmap(spearman_pval, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/Imagenet_Subsets/QuarterSpearman-pval.png")
# plt.clf()
# title = "Pearson Correlation on Performance Between Tasks of Quarter IN Equivalent Models"
# plt.title(title)
# ax = sns.heatmap(pearson, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/Imagenet_Subsets/QuarterPearson.png")
# plt.clf()
# title = "Pearson Correlation p-values on Performance Between Tasks of Quarter IN Equivalent Models"
# plt.title(title)
# ax = sns.heatmap(pearson_pval, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/Imagenet_Subsets/QuarterPearson-pval.png")
# plt.clf()
#
#
# data = pandas.read_csv("results.csv")
# # Select Quarter and Half Equivalent models
# data = data[(data["Updates"] == int(100*1e6))]
# tasks = ["Imagenet", "CalTech-101", "Pets", "PetsDetection", "dtd", "CIFAR-100", "SUN397", "Eurosat",
#          "CLEVERNumObjects", "THORNumSteps", "THORDepth", "NYUDepth", "NYUWalkable", "THORActionPrediction"]
# n = len(tasks)
# spearman = np.zeros((n,n))
# pearson = np.zeros((n,n))
# spearman_pval = np.zeros((n,n))
# pearson_pval = np.zeros((n,n))
# for i in range(n):
#     for j in range(n):
#         values_i = data[tasks[i]]
#         values_j = data[tasks[j]]
#         s, sp = scipy.stats.spearmanr(values_i, values_j)
#         p, pp = scipy.stats.pearsonr(values_i, values_j)
#         spearman[i][j] = s
#         pearson[i][j] = p
#         spearman_pval[i][j] = sp
#         pearson_pval[i][j] = pp
#
# plt.figure(figsize=(20, 20))
# title = "Spearman Correlation on Performance Between Tasks of Half IN Equivalent Models"
# plt.title(title)
# ax = sns.heatmap(spearman, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/Imagenet_Subsets/HalfSpearman.png")
# plt.clf()
# title = "Spearman Correlation p-values on Performance Between Tasks of Half IN Equivalent Models"
# plt.title(title)
# ax = sns.heatmap(spearman_pval, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/Imagenet_Subsets/HalfSpearman-pval.png")
# plt.clf()
# title = "Pearson Correlation on Performance Between Tasks of Half IN Equivalent Models"
# plt.title(title)
# ax = sns.heatmap(pearson, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/Imagenet_Subsets/HalfPearson.png")
# plt.clf()
# title = "Pearson Correlation p-values on Performance Between Tasks of Half IN Equivalent Models"
# plt.title(title)
# ax = sns.heatmap(pearson_pval, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/Imagenet_Subsets/HalfPearson-pval.png")
# plt.clf()

# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# swav_is_better = {}
# for task in ALL_TASKS:
#     if task == "Imagenetv2":
#         continue
#     swav_is_better[task] = data.loc["Supervised"][task] < data.loc["SWAV_800"][task]
#
# k = len([val for val in swav_is_better.values() if val])
# n = len(swav_is_better)
# p = 0.5
# print("SWAV is better in %d/%d tasks" % (k, n))
# print("the p-value is:", scipy.stats.binom_test(k, n=n, p=p, alternative='greater'))
#
#
# data = [
#     {"Encoder": "MoCov2Places", "KITTI": 0.5501},
#     {"Encoder": "MoCov2Kinetics", "KITTI": 0.5410},
#     {"Encoder": "MoCov2_800", "KITTI": 0.5404},
#     {"Encoder": "MoCov2Combination", "KITTI": 0.53902},
#     {"Encoder": "MoCov2_200", "KITTI": 0.495},
#     {"Encoder": "MoCov1_200", "KITTI": 0.4933},
#     {"Encoder": "SWAVPlaces", "KITTI": 0.481},
#     {"Encoder": "SWAVKinetics", "KITTI": 0.48},
#     {"Encoder": "MoCov2Taskonomy", "KITTI": 0.4723},
#     {"Encoder": "SWAV_800", "KITTI": 0.4657},
#     {"Encoder": "SWAV_200", "KITTI": 0.441},
#     {"Encoder": "Supervised", "KITTI": 0.4233},
#     {"Encoder": "SWAVTaskonomy", "KITTI": 0.4101},
#     {"Encoder": "PIRL", "KITTI": 0.402},
# ]
# data = pandas.DataFrame(data)
# sns.barplot(y="Encoder", x="KITTI", data=data)
# plt.show()


#
#
# ###### Number of data points vs performance
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# data = data.loc[["SWAV_200", "MoCov2_200", "SWAVHalfIN", "MoCov2HalfIN", "SWAVUnbalancedIN", "MoCov2UnbalancedIN",
#                  "SWAVQuarterIN", "MoCov2QuarterIN", "SWAVLogIN", "MoCov2LogIN"]]
# # data = data.drop(["TaskonomyDepth", "CityscapesSemanticSegmentation"], axis=1)
# sns.set_theme()
#
# n = len(ALL_TASKS)
# spearman = np.zeros((n,1))
# pearson = np.zeros((n,1))
# spearman_pval = np.zeros((n,1))
# pearson_pval = np.zeros((n,1))
# for i in range(n):
#     values_i = data[ALL_TASKS[i]]
#     values_j = data["DatasetSize"]
#     s, sp = scipy.stats.spearmanr(values_i, values_j)
#     p, pp = scipy.stats.pearsonr(values_i, values_j)
#     spearman[i] = s
#     pearson[i] = p
#     spearman_pval[i] = sp
#     pearson_pval[i] = pp
#
# heatmap = np.concatenate((spearman, spearman_pval, pearson, pearson_pval), axis=1)
#
# plt.figure(figsize=(15, 15))
# title = "Correlation Between Encoder Dataset Size and Test Performance"
# plt.title(title)
# ax = sns.heatmap(heatmap, annot=True)
# ax.set_yticklabels(ALL_TASKS, rotation=0)
# ax.set_xticklabels(["spearman", "spearman_pval", "pearson", "pearson_pval"], rotation=0)
# plt.savefig("graphs/dataset_size/correlations.png")
# plt.clf()
#
# ###### Number of epochs vs performance
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# data = data.loc[["SWAV_800", "SWAV_200", "SWAV_50", "MoCov2_800", "MoCov2_200", "MoCov2_100", "MoCov2_50",
#                  "SWAVHalfIN", "SWAVHalfIN_100", "MoCov2HalfIN", "MoCov2HalfIN_100",
#                  "SWAVUnbalancedIN", "SWAVUnbalancedIN_100", "MoCov2UnbalancedIN", "SWAVUnbalancedIN_100",
#                  "SWAVQuarterIN", "MoCov2QuarterIN", "SWAVLogIN", "MoCov2LogIN"]]
# # data = data.drop(["TaskonomyDepth", "CityscapesSemanticSegmentation"], axis=1)
# sns.set_theme()
#
# n = len(ALL_TASKS)
# spearman = np.zeros((n,1))
# pearson = np.zeros((n,1))
# spearman_pval = np.zeros((n,1))
# pearson_pval = np.zeros((n,1))
# for i in range(n):
#     values_i = data[ALL_TASKS[i]]
#     values_j = data["Updates"]
#     s, sp = scipy.stats.spearmanr(values_i, values_j)
#     p, pp = scipy.stats.pearsonr(values_i, values_j)
#     spearman[i] = s
#     pearson[i] = p
#     spearman_pval[i] = sp
#     pearson_pval[i] = pp
#
# heatmap = np.concatenate((spearman, spearman_pval, pearson, pearson_pval), axis=1)
#
# plt.figure(figsize=(15, 15))
# title = "Correlation Between Encoder Number of Updates and Test Performance"
# plt.title(title)
# ax = sns.heatmap(heatmap, annot=True)
# ax.set_yticklabels(ALL_TASKS, rotation=0)
# ax.set_xticklabels(["spearman", "spearman_pval", "pearson", "pearson_pval"], rotation=0)
# plt.savefig("graphs/num_updates/correlations.png")
# plt.clf()
#
###### Balanced vs performance
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# data = data.drop(["KineticsActionPrediction", "Imagenetv2"], axis=1)
# full_data = data
# data = data.loc[[#"SWAV_50", "MoCov2_50",
#                  #"SWAVHalfIN_100", "MoCov2HalfIN_100",
#                  "SWAVUnbalancedIN_100", "MoCov2UnbalancedIN_100",
#                  "SWAVQuarterIN", "MoCov2QuarterIN", "SWAVLogIN", "MoCov2LogIN"]]
# sns.set_theme()
#
# task_results = np.zeros((len(ALL_TASKS),len(data)))
# for t, task in enumerate(ALL_TASKS):
#     for e, encoder in enumerate(data.index):
#         task_results[t,e] = data.loc[encoder][task]
#     # task_results[t] -= min(full_data[task])
#     # task_results[t] /= (max(full_data[task]) - min(full_data[task]))
#     d = np.array((full_data[task]))
#     d = d[np.logical_not(np.isnan(d))]
#     task_results[t] -= np.mean(d)
#     task_results[t] /= np.std(d - np.mean(full_data[task]))
#
#
# log = []
# linear = []
# balanced = []
# for i in range(len(data.index)):
#     if "Log" in data.index[i]:
#         log.append(task_results[i])
#     elif "Unbalanced" in data.index[i]:
#         linear.append(task_results[i])
#     else:
#         balanced.append(task_results[i])
#
# log = np.concatenate(log, axis=0)
# linear = np.concatenate(linear, axis=0)
# balanced = np.concatenate(balanced, axis=0)
#
#
# data = [{"Dataset Balance": "log", "Score": d} for d in log] + \
#        [{"Dataset Balance": "linear", "Score": d} for d in linear] + \
#        [{"Dataset Balance": "balanced", "Score": d} for d in balanced]
# data = pandas.DataFrame(data)
# plt.title("Normalized End Task Test Results vs. Distribution of Encoder Dataset Sample")
# sns.violinplot(x="Dataset Balance", y="Score", data=data)
# plt.show()
#
# print("ANOVA")
# print("Log vs. Liner:", scipy.stats.f_oneway(log, linear))
# print("Log vs. Balanced:", scipy.stats.f_oneway(log, balanced))
# print("Liner vs. Balanced:", scipy.stats.f_oneway(linear, balanced))
# print("All:", scipy.stats.f_oneway(log, linear, balanced))
# print("Kruskal")
# print("Log vs. Liner:", scipy.stats.kruskal(log, linear))
# print("Log vs. Balanced:", scipy.stats.kruskal(log, balanced))
# print("Liner vs. Balanced:", scipy.stats.kruskal(linear, balanced))
# print("All:", scipy.stats.kruskal(log, linear, balanced))
#
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
#
# sun_datapoints = []
# taskonomy_datapoints = []
# kinetics_datapoints = []
# caltech_datapoints = []
#
# sun_mean = data.loc["Supervised"]["SUN397"]
# taskonomy_mean = data.loc["Supervised"]["TaskonomyDepth"]
# kinetics_mean = data.loc["Supervised"]["KineticsActionPrediction"]
# caltech_mean = data.loc["Supervised"]["CalTech-101"]
#
# for dp in data.index:
#     d = data.loc[dp]
#     if d["Dataset"] in ["HalfImagenet", "LogImagenet", "UnbalancedImagenet", "QuarterImageNet"]:
#         continue
#     sun_datapoints.append({"Score": d["SUN397"]-sun_mean, "Task": "SUN397", "Dataset": d["Dataset"], "Name": dp})
#     taskonomy_datapoints.append({"Score": d["TaskonomyDepth"]-taskonomy_mean, "Task": "Taskonomy Depth", "Dataset": d["Dataset"], "Name": dp})
#     kinetics_datapoints.append({"Score": d["KineticsActionPrediction"]-kinetics_mean, "Task": "Kinetics Action Prediction", "Dataset": d["Dataset"], "Name": dp})
#     caltech_datapoints.append({"Score": d["CalTech-101"]-caltech_mean, "Task": "CalTech101", "Dataset": d["Dataset"], "Name": dp})
# # sns.swarmplot(x="Task", y="Score", hue="Dataset", data=pandas.DataFrame(datapoints), s=10)
# sns.set_theme()
# plt.figure(1)
# sns.barplot(x="Name", y="Score", hue="Dataset", data=pandas.DataFrame(sun_datapoints).sort_values("Score"), dodge=False)
# plt.figure(2)
# sns.barplot(x="Name", y="Score", hue="Dataset", data=pandas.DataFrame(taskonomy_datapoints).sort_values("Score"), dodge=False)
# plt.figure(3)
# sns.barplot(x="Name", y="Score", hue="Dataset", data=pandas.DataFrame(kinetics_datapoints).sort_values("Score"), dodge=False)
# plt.figure(4)
# sns.barplot(x="Name", y="Score", hue="Dataset", data=pandas.DataFrame(caltech_datapoints).sort_values("Score"), dodge=False)
# plt.show()
#
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# datapoints = []
# TASKS = ["Imagenet", "Pets",
#          "CIFAR-100", "CalTech-101", "Eurosat", "dtd", "CityscapesSemanticSegmentation", "EgoHands",
#          "CLEVERNumObjects", "SUN397", "THORNumSteps", "THORActionPrediction", "THORDepth", "nuScenesActionPrediction", "NYUDepth", "NYUWalkable", "THORDepth", "KITTI",
#          "TaskonomyDepth", "KineticsActionPrediction"]
# for dp in data.index:
#     d = data.loc[dp]
#     if d["Updates"] != 200000000:
#         continue
#     if d["Dataset"] in ["HalfImagenet", "LogImagenet", "UnbalancedImagenet", "QuarterImageNet"]:
#         continue
#     if dp in ["SWAV_200", "MoCov2_200"]:
#         encoder = "Imagenet"
#     elif dp in ["SWAVPlaces", "MoCov2Places"]:
#         encoder = "Places"
#     else:
#         encoder = "Other"
#     for task in TASKS:
#         datapoints.append({"Normalize Score": d[task+"-normalized"], "Task": task, "Encoder": encoder})
# sns.scatterplot(
#     y="Task",
#     x="Normalize Score",
#     hue="Encoder",
#     size="Encoder",
#     sizes={"Imagenet": 100, "Places": 100, "Other": 30},
#     data=pandas.DataFrame(datapoints)
# )
# plt.show()



####### Classification Task vs IN Score
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# datapoints = []
# for encoder in data.index:
#     if data.loc[encoder]["Imagenet"] > 0.5:
#         for task in EMBEDDING_STRUCTURAL_TASKS:
#             datapoints.append({
#                 "Encoder": encoder,
#                 "Score": data.loc[encoder][task],
#                 "IN Score": data.loc[encoder]["Imagenet"],
#                 "TaskType": "Structural",
#                 "Task": task
#             })
#         # for task in EMBEDDING_SEMANTIC_TASKS:
#         #     datapoints.append({
#         #         "Encoder": encoder,
#         #         "Score": data.loc[encoder][task],
#         #         "IN Score": data.loc[encoder]["Imagenet"],
#         #         "TaskType": "Semantic",
#         #         "Task": task
#         #     })
# sns.set()
# plt.title("Structural Classification Task vs IN Score")
# sns.color_palette("bright")
# sns.lmplot(x="IN Score", y="Score", hue="Task", data=pandas.DataFrame(datapoints))
# plt.savefig("graphs/classification_vs_IN/Structural")
# plt.clf()
#
# datapoints = []
# for encoder in data.index:
#     if data.loc[encoder]["Imagenet"] > 0.5:
#         # for task in EMBEDDING_STRUCTURAL_TASKS:
#         #     datapoints.append({
#         #         "Encoder": encoder,
#         #         "Score": data.loc[encoder][task],
#         #         "IN Score": data.loc[encoder]["Imagenet"],
#         #         "TaskType": "Structural",
#         #         "Task": task
#         #     })
#         for task in EMBEDDING_SEMANTIC_TASKS:
#             datapoints.append({
#                 "Encoder": encoder,
#                 "Score": data.loc[encoder][task],
#                 "IN Score": data.loc[encoder]["Imagenet"],
#                 "TaskType": "Semantic",
#                 "Task": task
#             })
# sns.set()
# plt.title("Semantic Classification Task vs IN Score")
# sns.color_palette("bright")
# sns.lmplot(x="IN Score", y="Score", hue="Task", data=pandas.DataFrame(datapoints))
# plt.savefig("graphs/classification_vs_IN/Semantic")
# plt.clf()


####### Make Correlation Plots for Similar Datasets
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# fig, axes = plt.subplots(4, 1, figsize=(8, 8))
# fig.suptitle("End Task Performance on Different End Tasks")
# # CalTech
# datapoints = []
# for encoder in data.index:
#     if data.loc[encoder]["Imagenet"] > 0.4:
#         datapoints.append({
#             "Encoder": encoder,
#             "Score": data.loc[encoder]["CalTech-101"],
#             "IN Score": data.loc[encoder]["Imagenet"],
#             "Dataset": "Imagenet" if data.loc[encoder]["Dataset"] == "Imagenet" else "Other"
#         })
# sns.set()
# sns.color_palette("bright")
# datapoints.sort(key=lambda x: x["Score"])
# ax = sns.barplot(x="Encoder", y="Score", hue="Dataset", dodge=False, data=pandas.DataFrame(datapoints), ax=axes[0])
# ax.set_ylabel("Top 1 Accuracy")
# ax.get_xaxis().set_visible(False)
# ax.set_title("CalTech Classification Top-1 Accuracy")
# # # SUN397
# datapoints = []
# for encoder in data.index:
#     if data.loc[encoder]["Imagenet"] > 0.4:
#         datapoints.append({
#             "Encoder": encoder,
#             "Score": data.loc[encoder]["SUN397"],
#             "IN Score": data.loc[encoder]["Imagenet"],
#             "Dataset": "Places" if data.loc[encoder]["Dataset"] == "Places" else "Other"
#         })
# sns.set()
# plt.title("SUN397 vs Imagenet Score")
# sns.color_palette("bright")
# datapoints.sort(key=lambda x: x["Score"])
# ax = sns.barplot(x="Encoder", y="Score", hue="Dataset", dodge=False, data=pandas.DataFrame(datapoints), ax=axes[1])
# ax.set_ylabel("Top 1 Accuracy")
# ax.get_xaxis().set_visible(False)
# ax.set_title("SUN397 Classification Top-1 Accuracy")
# # Kinetics
# datapoints = []
# for encoder in data.index:
#     if data.loc[encoder]["Imagenet"] > 0.4:
#         datapoints.append({
#             "Encoder": encoder,
#             "Score": data.loc[encoder]["KineticsActionPrediction"],
#             "IN Score": data.loc[encoder]["Imagenet"],
#             "Dataset": "Kinetics" if data.loc[encoder]["Dataset"] == "Kinetics" else "Other"
#         })
# sns.set()
# sns.color_palette("bright")
# datapoints.sort(key=lambda x: x["Score"])
# ax = sns.barplot(x="Encoder", y="Score", hue="Dataset", dodge=False, data=pandas.DataFrame(datapoints), ax=axes[2])
# ax.set_ylabel("Top 1 Accuracy")
# ax.get_xaxis().set_visible(False)
# ax.set_title("Kinetics Classification Top-1 Accuracy")
# ## Taskonomy
# datapoints = []
# for encoder in data.index:
#     if data.loc[encoder]["Imagenet"] > 0.25:
#         datapoints.append({
#             "Encoder": encoder,
#             "Score": data.loc[encoder]["TaskonomyDepth"],
#             "IN Score": data.loc[encoder]["Imagenet"],
#             "Dataset": "Taskonomy" if data.loc[encoder]["Dataset"] == "Taskonomy" else "Other"
#         })
# sns.set()
# sns.color_palette("bright")
# datapoints.sort(key=lambda x: x["Score"])
# ax = sns.barplot(x="Encoder", y="Score", hue="Dataset", dodge=False, data=pandas.DataFrame(datapoints), ax=axes[3])
# ax.set_ylabel("L1 Error")
# ax.get_xaxis().set_visible(False)
# ax.set_title("TaskonomyDepth L1 Error")
# plt.savefig("graphs/datasets_with_similar_end_tasks/similar_tasks")


####### Make nuScenes Human Performance Graph
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# # CalTech
# datapoints = []
# for encoder in data.index:
#     if data.loc[encoder]["Imagenet"] > 0.4:
#         datapoints.append({
#             "Encoder": encoder,
#             "Score": data.loc[encoder]["nuScenesActionPrediction"],
#             "Type": "Model"
#         })
# datapoints.append({"Encoder": "Roozbeh", "Score": 0.725, "Type": "Human"})
# datapoints.append({"Encoder": "Klemen", "Score": 0.84, "Type": "Human"})
# datapoints.append({"Encoder": "Gabriel", "Score": 0.9, "Type": "Human"})
# datapoints.sort(key=lambda x: x["Score"], reverse=True)
# sns.set()
# plt.title("nuScenes Action Prediction Score")
# sns.color_palette("bright")
# sns.barplot(x="Score", y="Encoder", hue="Type", data=pandas.DataFrame(datapoints), dodge=False)
# plt.show()
# plt.savefig("graphs/datasets_with_similar_end_tasks/CalTech")
# plt.clf()

#
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# # CalTech
# datapoints = []
# POINTS = [
#     "Imagenet",
#     "Pets",
#     "CIFAR-100",
#     "CalTech-101",
#     "Eurosat",
#     "dtd",
#     "SUN397",
#     "KineticsActionPrediction",
#     "CLEVERNumObjects",
#     "THORNumSteps",
#     "THORActionPrediction",
#     "nuScenesActionPrediction"
# ]
# for task in POINTS:
#     datapoints.append({
#         "Task": task,
#         "Score": abs(data.loc["SWAV_200"][task] - data.loc["SWAV_200_2"][task]),
#     })
# datapoints.sort(key=lambda x: x["Score"], reverse=True)
# sns.set()
# plt.title("nuScenes Action Prediction Score")
# sns.color_palette("bright")
# sns.barplot(x="Score", y="Task", data=pandas.DataFrame(datapoints), dodge=False)
# plt.show()
# plt.savefig("graphs/datasets_with_similar_end_tasks/CalTech")
# plt.clf()

# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# points = []
# for encoder in data.index:
#     dp = dict(data.loc[encoder])
#     dp["Encoder"] = encoder
#     points.append(dp)
# top_spots = {}
# for task in ALL_TASKS:
#     points.sort(key=lambda x: x[task])
#     encoder = points[-1]['Encoder']
#     if encoder not in top_spots:
#         top_spots[encoder] = []
#     top_spots[encoder].append(task)
#
# for task, spot in top_spots.items():
#     print(task, spot)
#
# count_swav_better = 0
# for task in ALL_TASKS:
#     if data.loc["SWAVPlaces"][task] > data.loc["Supervised"][task]:
#         count_swav_better += 1
# print(count_swav_better)



# ##### Make Per Dataset Violin Plot of Scores for MoCo and SWAV 200
# data = pandas.read_csv("results.csv")
# ms_data = data[(data["Method"] == "MoCo") | (data["Method"] == "SWAV")]
# ms_200_data = ms_data[ms_data["Epochs"] == 200]
# ms_200_full_data = ms_200_data[ms_200_data["Updates"] == int(1e6 * 200)]
#
# sns.set_theme()
# colors = sns.color_palette()
# # palette = {method: colors[i] for i, method in enumerate(set(data["Dataset"]))}
# plt.figure(figsize=(10, 5))
#
# ee = ms_200_full_data.set_index("Encoder").drop("MoCov1_200")
# per_task_ms_200_full = []
# for task in PIXELWISE_SEMANTIC_TASKS + EMBEDDING_SEMANTIC_TASKS + PIXELWISE_STRUCTURAL_TASKS + EMBEDDING_STRUCTURAL_TASKS:
#     for encoder in ee.index:
#         if encoder == "SWAVKinetics" and task == "THORDepth":
#             continue
#
#         if task in EMBEDDING_SEMANTIC_TASKS:
#             ttype = "Embedding-Semantic"
#         elif task in EMBEDDING_STRUCTURAL_TASKS:
#             ttype = "Embedding-Structural"
#         elif task in PIXELWISE_SEMANTIC_TASKS:
#             ttype = "Pixelwise-Semantic"
#         elif task in PIXELWISE_STRUCTURAL_TASKS:
#             ttype = "Pixelwise-Structural"
#         else:
#             ttype = "Other"
#
#         per_task_ms_200_full.append({
#             "Encoder": encoder,
#             "Task": task,
#             "Method": ee.loc[encoder, "Method"],
#             "Dataset": ee.loc[encoder, "Dataset"],
#             "NormalizedScore": ee.loc[encoder, task+"-normalized"],
#             "TaskType": ttype
#         })
#
# plt.title("Normalized Per Dataset Performance of SWAV and MoCov2 Encoders trained on 1.3M Image Datasets")
# df = pandas.DataFrame(per_task_ms_200_full)
# # order = ["Embedding-Semantic", "Pixelwise-Semantic", "Embedding-Structural", "Pixelwise-Structural"]
# ax = sns.violinplot(x="Dataset", y="NormalizedScore", hue="Dataset", data=df, dodge=False)
# plt.show()
# # plt.savefig("graphs/mocov2_and_swav_200_different_task_types/violin.png", dpi=100)
# # plt.clf()
#
#
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# # CalTech
# datapoints = []
# for encoder in data.index:
#     if data.loc[encoder]["Imagenet"] < 0.4:
#         print(encoder)
#     for task in ALL_TASKS:
#         if data.loc[encoder]["Imagenet"] > 0.0:
#             if task in EMBEDDING_SEMANTIC_TASKS:
#                 ttype = "Semantic"
#                 otype = "Embedding"
#             elif task in EMBEDDING_STRUCTURAL_TASKS:
#                 ttype = "Structural"
#                 otype = "Embedding"
#             elif task in PIXELWISE_SEMANTIC_TASKS:
#                 ttype = "Semantic"
#                 otype = "Pixelwise"
#             elif task in PIXELWISE_STRUCTURAL_TASKS:
#                 ttype = "Structural"
#                 otype = "Pixelwise"
#             datapoints.append({
#                 "Encoder": encoder,
#                 "Task": task,
#                 "TaskType": ttype,
#                 "OutputType": otype,
#                 "Score": data.loc[encoder][task],
#                 "IN Score": data.loc[encoder]["Imagenet"],
#                 "Dataset": "Imagenet" if data.loc[encoder]["Dataset"] == "Imagenet" else "Other"
#             })
# sns.set()
# sns.color_palette("bright")
# datapoints.sort(key=lambda x: x["Score"])
# ax = sns.lmplot(x="IN Score", y="Score", hue="Task", row="TaskType", col="OutputType", data=pandas.DataFrame(datapoints))
# # ax.set_ylabel("Score")
# # ax.get_xaxis().set_visible(False)
# # ax.set_title("CalTech Classification Top-1 Accuracy")
# plt.show()
#
#
# make_csv()
# sns.set()
# sns.color_palette("bright")
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# fig, axes = plt.subplots(4, 1)
# fig.suptitle("Similar Datasets")
#
# datapoints = []
# for encoder in data.index:
#     datapoints.append({
#         "Encoder": encoder,
#         "Score": data.loc[encoder]["CalTech-101"],
#         "Dataset": "Imagenet" if data.loc[encoder]["Dataset"] == "Imagenet" else "Other"
#     })
# ax = sns.kdeplot(x="Score", hue="Dataset", data=pandas.DataFrame(datapoints), ax=axes[0], fill=True,
#                   common_norm=False, bw_adjust=0.8)
# datapoints = []
# for encoder in data.index:
#     datapoints.append({
#         "Encoder": encoder,
#         "Score": data.loc[encoder]["KineticsActionPrediction"],
#         "Dataset": "Kinetics" if data.loc[encoder]["Dataset"] == "Kinetics" else "Other"
#     })
# ax = sns.kdeplot(x="Score", hue="Dataset", data=pandas.DataFrame(datapoints), ax=axes[1], fill=True,
#                   common_norm=False, bw_adjust=0.8)
# datapoints = []
# for encoder in data.index:
#     datapoints.append({
#         "Encoder": encoder,
#         "Score": data.loc[encoder]["SUN397"],
#         "Dataset": "Places" if data.loc[encoder]["Dataset"] == "Places" else "Other"
#     })
# ax = sns.kdeplot(x="Score", hue="Dataset", data=pandas.DataFrame(datapoints), ax=axes[2], fill=True,
#                   common_norm=False, bw_adjust=0.8)
# datapoints = []
# for encoder in data.index:
#     datapoints.append({
#         "Encoder": encoder,
#         "Score": data.loc[encoder]["TaskonomyDepth"],
#         "Dataset": "Taskonomy" if data.loc[encoder]["Dataset"] == "Taskonomy" else "Other"
#     })
# ax = sns.kdeplot(x="Score", hue="Dataset", data=pandas.DataFrame(datapoints), ax=axes[3], fill=True,
#                   common_norm=False, bw_adjust=0.8)
# plt.show()



######## Figure 7. Performance of Encoders with Similar Datasets
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data[data["Epochs"] == 200]
# data = data[data["Updates"] == int(1e6 * 200)]
# data = data.set_index("Encoder")
# data = data.drop("MoCov1_200")
# data = data.drop("SWAV_200_2")
# # data = data.drop("Supervised")
# data = data.drop("Random")
# pastels = sns.color_palette("pastel")
#
# sns.set()
# sns.color_palette("bright")
# sns.set_theme(style="whitegrid", font_scale=1.8)
# fig, axes = plt.subplots(1, 4, figsize=(12, 8))
# # fig.suptitle("Performance of Encoders on End Tasks With Similar Datasets")
#
# datapoints = []
# for encoder in data.index:
#     if not math.isnan(data.loc[encoder]["CalTech-101"]):
#         if data.loc[encoder]["Dataset"] == "Imagenet":
#             ds = "ImageNet"
#         elif data.loc[encoder]["Dataset"] == "Combination":
#             ds = "Combination"
#         else:
#             ds = "Other"
#         datapoints.append({
#             "Encoder": encoder,
#             "Score": data.loc[encoder]["CalTech-101"],
#             "Encoder Dataset": ds
#         })
# datapoints.sort(key=lambda x: x["Score"], reverse=True)
# ax = sns.barplot(y="Encoder", x="Score", dodge=False, hue="Encoder Dataset", data=pandas.DataFrame(datapoints), ax=axes[0],
#                  palette={"ImageNet": pastels[2],  "Combination": pastels[4], "Other": pastels[7]},
#                  hue_order=["ImageNet", "Combination", "Other"])
# ax.set_title("CalTech Cls.")
# # ax.get_yaxis().set_visible(False)
# ax.set_yticks([])
# ax.set_ylabel("Performance of Encoders on End Task")
# ax.set_xlabel("Top-1 Accuracy")
# # ax.legend(loc='lower center', prop={'size': 17})
# ax.legend(bbox_to_anchor=(1.05, -0.13),borderaxespad=0, prop={'size': 17})
#
# # datapoints = []
# # for encoder in data.index:
# #     if not math.isnan(data.loc[encoder]["nuScenesActionPrediction"]):
# #         if data.loc[encoder]["Dataset"] == "Kinetics":
# #             ds = "Kinetics"
# #         elif data.loc[encoder]["Dataset"] == "Combination":
# #             ds = "Combination"
# #         else:
# #             ds = "Other"
# #         datapoints.append({
# #             "Encoder": encoder,
# #             "Score": data.loc[encoder]["nuScenesActionPrediction"],
# #             "Encoder Dataset": ds
# #         })
# # datapoints.sort(key=lambda x: x["Score"], reverse=True)
# # ax = sns.barplot(y="Encoder", x="Score", dodge=False, hue="Encoder Dataset", data=pandas.DataFrame(datapoints), ax=axes[1],
# #                  palette={"Kinetics": pastels[2], "Combination": pastels[4], "Other": pastels[7]},
# #                  hue_order=["Kinetics", "Combination", "Other"])
# # ax.set_title("nuScenes")
# # ax.get_yaxis().set_visible(False)
# # ax.set_xlabel("Top-1 Accuracy")
# # ax.legend(loc='lower center')
#
# datapoints = []
# for encoder in data.index:
#     if not math.isnan(data.loc[encoder]["KineticsActionPrediction"]):
#         if data.loc[encoder]["Dataset"] == "Kinetics":
#             ds = "Kinetics"
#         elif data.loc[encoder]["Dataset"] == "Combination":
#             ds = "Combination"
#         else:
#             ds = "Other"
#         datapoints.append({
#             "Encoder": encoder,
#             "Score": data.loc[encoder]["KineticsActionPrediction"],
#             "Encoder Dataset": ds
#         })
# datapoints.sort(key=lambda x: x["Score"], reverse=True)
# ax = sns.barplot(y="Encoder", x="Score", dodge=False, hue="Encoder Dataset", data=pandas.DataFrame(datapoints), ax=axes[1],
#                  palette={"Kinetics": pastels[2], "Combination": pastels[4], "Other": pastels[7]},
#                  hue_order=["Kinetics", "Combination", "Other"])
# ax.set_title("Kinetics Act. Pred.")
# ax.get_yaxis().set_visible(False)
# ax.set_xlabel("Top-1 Accuracy")
# # ax.legend(loc='lower center', prop={'size': 17})
# ax.legend(bbox_to_anchor=(1.05, -0.13),borderaxespad=0, prop={'size': 17})
#
# datapoints = []
# for encoder in data.index:
#     if not math.isnan(data.loc[encoder]["SUN397"]):
#         if data.loc[encoder]["Dataset"] == "Places":
#             ds = "Places"
#         elif data.loc[encoder]["Dataset"] == "Combination":
#             ds = "Combination"
#         else:
#             ds = "Other"
#         datapoints.append({
#             "Encoder": encoder,
#             "Score": data.loc[encoder]["SUN397"],
#             "Encoder Dataset": ds
#         })
# datapoints.sort(key=lambda x: x["Score"], reverse=True)
# ax = sns.barplot(y="Encoder", x="Score", dodge=False, hue="Encoder Dataset", data=pandas.DataFrame(datapoints), ax=axes[2],
#                  palette={"Places": pastels[2], "Combination": pastels[4], "Other": pastels[7]},
#                  hue_order=["Places", "Combination", "Other"])
# ax.set_title("SUN Scene Cls.")
# ax.get_yaxis().set_visible(False)
# ax.set_xlabel("Top-1 Accuracy")
# # ax.legend(loc='lower center', prop={'size': 17})
# ax.legend(bbox_to_anchor=(1.05, -0.13),borderaxespad=0, prop={'size': 17})
#
# datapoints = []
# for encoder in data.index:
#     if not math.isnan(data.loc[encoder]["TaskonomyDepth"]):
#         if data.loc[encoder]["Dataset"] == "Taskonomy":
#             ds = "Taskonomy"
#         elif data.loc[encoder]["Dataset"] == "Combination":
#             ds = "Combination"
#         else:
#             ds = "Other"
#         datapoints.append({
#             "Encoder": encoder,
#             "Score": data.loc[encoder]["TaskonomyDepth"],
#             "Encoder Dataset": ds
#         })
# datapoints.sort(key=lambda x: x["Score"], reverse=True)
# ax = sns.barplot(y="Encoder", x="Score", dodge=False, hue="Encoder Dataset", data=pandas.DataFrame(datapoints), ax=axes[3],
#                  palette={"Taskonomy": pastels[2], "Combination": pastels[4], "Other": pastels[7]},
#                  hue_order=["Taskonomy", "Combination", "Other"])
# ax.set_title("Taskonomy Depth")
# ax.get_yaxis().set_visible(False)
# ax.set_xlabel("Negative L1 Error")
# # ax.legend(loc='lower center', prop={'size': 17})
# ax.legend(bbox_to_anchor=(1.05, -0.13),borderaxespad=0, prop={'size': 17})

# datapoints = []
# for encoder in data.index:
#     if not math.isnan(data.loc[encoder]["NYUDepth"]):
#         if data.loc[encoder]["Dataset"] == "Taskonomy":
#             ds = "Taskonomy"
#         elif data.loc[encoder]["Dataset"] == "Combination":
#             ds = "Combination"
#         else:
#             ds = "Other"
#         datapoints.append({
#             "Encoder": encoder,
#             "Score": data.loc[encoder]["NYUDepth"],
#             "Encoder Dataset": ds
#         })
# datapoints.sort(key=lambda x: x["Score"], reverse=True)
# ax = sns.barplot(y="Encoder", x="Score", dodge=False, hue="Encoder Dataset", data=pandas.DataFrame(datapoints), ax=axes[3],
#                  palette={"Taskonomy": pastels[2], "Combination": pastels[4], "Other": pastels[7]},
#                  hue_order=["Taskonomy", "Combination", "Other"])
# ax.set_title("NYU Depth")
# ax.get_yaxis().set_visible(False)
# ax.set_xlabel("L1 Error")
# ax.legend(loc='lower center')

# plt.show()
# plt.savefig("graphs/datasets_with_similar_end_tasks/similar_tasks.pdf", bbox_inches='tight')


####### Figure 4. Omni
make_csv()
data = pandas.read_csv("results.csv")
data = data.set_index("Encoder")
# CalTech
datapoints = []
for encoder in data.index:
    for task in PIXELWISE_SEMANTIC_TASKS + PIXELWISE_STRUCTURAL_TASKS + EMBEDDING_SEMANTIC_TASKS + EMBEDDING_STRUCTURAL_TASKS:
        if data.loc[encoder]["Imagenet"] > 0.4:
            if task in EMBEDDING_SEMANTIC_TASKS:
                ttype = "Semantic Image-level"
            elif task in EMBEDDING_STRUCTURAL_TASKS:
                ttype = "Structural Image-level"
            elif task in PIXELWISE_SEMANTIC_TASKS:
                ttype = "Semantic Pixelwise"
            elif task in PIXELWISE_STRUCTURAL_TASKS:
                ttype = "Structural  Pixelwise"
            datapoints.append({
                "Encoder": encoder,
                "Task": REAL_NAMES[task],
                "TaskType": ttype,
                "Score": data.loc[encoder][task],
                "ImageNet Accuracy": data.loc[encoder]["Imagenet"],
                "Dataset": "Imagenet" if data.loc[encoder]["Dataset"] == "Imagenet" else "Other",
                "NormalizedScore": data.loc[encoder, task + "-normalized"],
            })
# sns.set()
pal = sns.color_palette("deep", 21)
sns.set_theme(style="whitegrid", font_scale=2)
# datapoints.sort(key=lambda x: x["Score"])
lm = sns.lmplot(
    x="ImageNet Accuracy",
    y="Score",
    col="TaskType",
    hue="Task",
    sharex=True,
    sharey=False,
    legend_out=True,
    height=10,
    aspect=0.7,
    palette=pal,
    scatter_kws={"s": 100},
    markers=[ "^", "^", "^", "X", "X", "X", "X", "X", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "o", "s", "s", "s", "s"],
    data=pandas.DataFrame(datapoints),
)
fig = lm.fig

fig.axes[0].set_title("Pixelwise Semantic")
fig.axes[0].set_ylabel("mIOU")

fig.axes[1].set_title("Pixelwise Structural")
fig.axes[1].set_ylabel("Negative L1 Error                                       mIOU")

fig.axes[2].set_title("Image-Level Semantic")
fig.axes[2].set_ylabel("Top-1 Accuracy")

fig.axes[3].set_title("Image-Level Structural")
fig.axes[3].set_ylabel("Top-1 Accuracy")

# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=21, mode="expand", borderaxespad=0.)
# ax.set_ylabel("Score")
# ax.get_xaxis().set_visible(False)
# ax.set_title("CalTech Classification Top-1 Accuracy")
plt.savefig("graphs/in_vs_tasks/omni.pdf", bbox_inches='tight')
# plt.show()


######### Figure 3. Percentage improvement of Best Encoder over Supervised ImageNet
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder")
# # CalTech
# datapoints = []
# table = []
# for encoder in data.index:
#     if encoder == "Supervised":
#         continue
#     for task in ALL_TASKS:
#         if not math.isnan(data.loc[encoder][task]):
#             datapoints.append({
#                 "Encoder": encoder,
#                 "Task": REAL_NAMES[task],
#                 "Score": data.loc[encoder][task],
#                 "SupervisedScore": data.loc["Supervised"][task],
#                 "Dataset": data.loc[encoder]["Dataset"],
#                 "Method": data.loc[encoder]["Method"],
#                 "NormalizedScore": data.loc[encoder, task + "-normalized"],
#             })
# for task in ALL_TASKS:
#     if task in REVERSED_SUCCESS_TASKS:
#         c = -1.0
#     else:
#         c= 1.0
#     task = REAL_NAMES[task]
#     tdps = [p for p in datapoints if p["Task"] == task]
#     tdps.sort(key=lambda x: x["Score"], reverse=True)
#     table.append({
#         "Task": task,
#         "Score": tdps[0]["Score"],
#         "Method": tdps[0]["Method"],
#         "Supervised Score": tdps[0]["SupervisedScore"],
#         "Performance Improvement": 100 * c * (tdps[0]["Score"] - tdps[0]["SupervisedScore"]) / (tdps[0]["SupervisedScore"]),
#         "Best Method": tdps[0]["Method"],
#         "Best Dataset": tdps[0]["Dataset"].replace("Log", "").replace("Quarter", "")
#     })
# table.sort(key=lambda x: x["Performance Improvement"])
# df = pandas.DataFrame(table)
# # df.to_csv("best_encoder_per_task.csv")
#
# sns.set(font_scale=10)
# sns.color_palette("bright")
# datapoints.sort(key=lambda x: x["Score"])
# sns.set_theme(style="whitegrid", font_scale=1.8)
#
# # plt.figure(figsize=(8, 8))
# fig, ax1 = plt.subplots(figsize=(8, 10))
# sns.barplot(x="Performance Improvement", y="Task", hue="Best Dataset", dodge=False, data=df, palette="pastel", ax=ax1)
# ax1.set_ylabel("End Task")
# ax1.set_xlabel("Percent Relative Performance Improvement\nOver Supervised Baseline")
# # ax2 = ax1.twinx()
#
# table = []
# for task in ALL_TASKS:
#     if task in REVERSED_SUCCESS_TASKS:
#         c = -1.0
#     else:
#         c= 1.0
#     task = REAL_NAMES[task]
#     tdps = [p for p in datapoints if p["Task"] == task]
#     tdps.sort(key=lambda x: x["Score"], reverse=True)
#     score = (100 * c * ((tdps[0]["Score"] - tdps[0]["SupervisedScore"]) / (tdps[0]["SupervisedScore"])))
#     table.append({
#         "Task": task,
#         "Score": tdps[0]["Score"],
#         "Method": "MoCov2" if tdps[0]["Method"] == "MoCo" else "SwAV",
#         "Supervised Score": tdps[0]["SupervisedScore"],
#         "Performance Improvement": score + 1 if score > 0 else score - 1,
#         "Best Dataset": tdps[0]["Dataset"].replace("Log", "").replace("Quarter", "")
#     })
# table.sort(key=lambda x: x["Performance Improvement"])
# df = pandas.DataFrame(table)
#
# # ax2 = sns.scatterplot(x="Performance Improvement", y="Task", style="Method", data=df, ax=ax2, s=50)
# # ax2.legend(title="   Best Method   ", loc=[0.59, 0.54])
# # ax2.set_ylim(ax1.get_ylim())
# # ax2.get_yaxis().set_visible(False)
# # ax.legend(title="Best Encoder Method\nand Training Set")
# # ax2.set_ylabel("End Task")
#
#
# # ax.get_xaxis().set_visible(False)
# # ax.set_title("CalTech Classification Top-1 Accuracy")
# plt.savefig("graphs/best_encoder/best_encoder.pdf", bbox_inches='tight')
# # plt.show()

#### Generating Pearson and Spearman Correlations for the relation between cka realtions and task scores
# make_csv()
# data = pandas.read_csv("results.csv")
# tasks = ["Imagenet", "Imagenetv2", "CalTech-101", "Pets", "dtd", "CIFAR-100", "SUN397", "Eurosat",
#          "EgoHands", "CityscapesSemanticSegmentation", "PetsDetection",
#          "THORNumSteps", "CLEVERNumObjects", "nuScenesActionPrediction", "THORActionPrediction",
#          "TaskonomyDepth", "NYUWalkable", "NYUDepth", "THORDepth",]
#
# n = len(tasks)
# m = 15
#
# encoders = []
# cka = []
# for f in glob.glob("graphs/cka/multi_model_layer_wise/ImageNet/*.npy"):
#     name = f.split("/")[-1].replace(".npy", "")
#     if name.split("-")[0] != name.split("-")[1]:
#         continue
#     name = name.split("-")[0]
#     if name in ["SWAV_100", "SWAV_200_2"]:
#         continue
#     cka.append(np.load(f))
#     encoders.append(name)
# cka = np.stack(cka, axis=2)
# cka = cka[np.array([0,0,0,0,0,1,1,1,1,2,2,2,3,3,4]), np.array([1,2,3,4,5,2,3,4,5,3,4,5,4,5,5])]
#
# data = data.set_index("Encoder")
# data = data.loc[encoders]
#
# spearman = np.zeros((n,m))
# pearson = np.zeros((n,m))
# spearman_pval = np.zeros((n,m))
# pearson_pval = np.zeros((n,m))
#
# for i in range(n):
#     for j in range(m):
#         values_i = data[tasks[i]]
#         values_j = cka[j]
#         s, sp = scipy.stats.spearmanr(values_i, values_j)
#         p, pp = scipy.stats.pearsonr(values_i, values_j)
#         spearman[i][j] = s
#         pearson[i][j] = p
#         spearman_pval[i][j] = sp
#         pearson_pval[i][j] = pp
#
# tasks = [t.replace("ActionPrediction", "").replace("SemanticSegmentation", "") for t in tasks]
# # fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# # fig.suptitle("Correlation of End Task Performance and CKA Values")
# ax = sns.heatmap(pearson, annot=True)
# ax.set_title("Spearman")
# # ax = sns.heatmap(spearman, annot=True, ax=axes[1])
# # ax.set_title("Pearson")
# # ax.set_xticks([])
# # ax.set_yticks([])
# # ax = sns.heatmap(pearson, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# corr_names = ["c-1", "c-2", "c-3", "c-4", "c-e", "1-2", "1-3", "1-4", "1-e", "2-3", "2-4", "2-e", "3-4", "3-e", "4-e"]
# ax.set_xticklabels(corr_names, rotation=30, rotation_mode="anchor", ha='right', va="center")
#
# plt.show()
#

###### Figure 8.
# make_csv()
# data = pandas.read_csv("results.csv")
# # data = data[data["Method"] == "SWAV"]
# # data = data[data["Epochs"] == 200]
# data = data[data["Updates"] == int(1e6 * 50)]
# data = data[data["Epochs"] == 200]
# data = data.set_index("Encoder")
#
# datapoints = []
# for encoder in data.index:
#     for task in ALL_TASKS:
#         if not math.isnan(data.loc[encoder][task]):
#             if task in EMBEDDING_SEMANTIC_TASKS:
#                 ttype = "Semantic Image-level"
#             elif task in EMBEDDING_STRUCTURAL_TASKS:
#                 ttype = "Structural Image-level"
#             elif task in PIXELWISE_SEMANTIC_TASKS:
#                 ttype = "Semantic Pixelwise"
#             elif task in PIXELWISE_STRUCTURAL_TASKS:
#                 ttype = "Structural  Pixelwise"
#             datapoints.append({
#                 "Encoder": encoder,
#                 "Task": REAL_NAMES[task],
#                 "End Task Score": data.loc[encoder][task],
#                 "ImageNet Score": data.loc[encoder]["Imagenet"],
#                 "Balance": data.loc[encoder]["Balance"],
#                 # "Supervised Score": data.loc["Supervised"][task],
#                 "Dataset": data.loc[encoder]["Dataset"],
#                 "Method": data.loc[encoder]["Method"],
#                 "Normalized Score": data.loc[encoder, task + "-normalized"],
#                 "Task Type": ttype
#             })
#
# differences = []
# for task in REAL_NAMES.values():
#     log_means = []
#     balanced_means = []
#     for dp in datapoints:
#         if dp["Task"] == task:
#             if dp["Balance"] == "Log":
#                 log_means.append(dp["End Task Score"])
#             if dp["Balance"] == "Balanced":
#                 balanced_means.append(dp["End Task Score"])
#             ttype = dp["Task Type"]
#     print(task, np.mean(log_means), np.mean(balanced_means))
#     differences.append({
#         "Percentage Improvement of Log-Unbalanced vs. Balanced": (np.mean(log_means) - np.mean(balanced_means)) / np.mean(balanced_means),
#         "Task": task,
#         "Task Type": ttype}
#     )
#
# # datapoints.sort(key=lambda x: x["Score"])
# df = pandas.DataFrame(datapoints)
#
# sns.set(font_scale=10)
# sns.color_palette("bright")
# sns.set_theme(style="whitegrid", font_scale=1.8)
#
#
# plt.figure(figsize=(8, 8))
# # differences.sort(key=lambda x: x["Difference of Means"])
# plt.title("Percentage Improvement of Encoders Trained on\nLog-Unbalanced ImageNet vs. Encoders Trained on Balanced ImageNet")
# ax = sns.barplot(x="Percentage Improvement of Log-Unbalanced vs. Balanced", y="Task", hue="Task Type", dodge=False, palette="pastel", data=pandas.DataFrame(differences))
# ax.legend(loc='lower right')
# ax = sns.scatterplot(y="Task", x="End Task Score", hue="Balance", data=df, palette="pastel")
# ax.set_xlabel("Percent Performance Improvement Over Supervised ImageNet")
# ax2 = ax.twinx()
# ax2.set_ylim(ax.get_ylim())
# ax2.set_yticks(np.arange(len(table)))
# ax2.set_yticklabels([t["Method"] for t in table])

# ax.get_xaxis().set_visible(False)
# ax.set_title("CalTech Classification Top-1 Accuracy")
# plt.savefig("graphs/IN_balance/log_vs_balanced.pdf", bbox_inches='tight')
#
#
#
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data[data["Updates"] == int(1e6 * 100)]
# data = data[data["Epochs"] == 200]
# data = data.set_index("Encoder")
#
# datapoints = []
# for encoder in data.index:
#     for task in ALL_TASKS:
#         if not math.isnan(data.loc[encoder][task]):
#             if task in EMBEDDING_SEMANTIC_TASKS:
#                 ttype = "Semantic Image-level"
#             elif task in EMBEDDING_STRUCTURAL_TASKS:
#                 ttype = "Structural Image-level"
#             elif task in PIXELWISE_SEMANTIC_TASKS:
#                 ttype = "Semantic Pixelwise"
#             elif task in PIXELWISE_STRUCTURAL_TASKS:
#                 ttype = "Structural  Pixelwise"
#             datapoints.append({
#                 "Encoder": encoder,
#                 "Task": REAL_NAMES[task],
#                 "End Task Score": data.loc[encoder][task],
#                 "ImageNet Score": data.loc[encoder]["Imagenet"],
#                 "Balance": data.loc[encoder]["Balance"],
#                 "Dataset": data.loc[encoder]["Dataset"],
#                 "Method": data.loc[encoder]["Method"],
#                 "Normalized Score": data.loc[encoder, task + "-normalized"],
#                 "Task Type": ttype
#             })
#
# differences = []
# for task in REAL_NAMES.values():
#     linear_means = []
#     balanced_means = []
#     for dp in datapoints:
#         if dp["Task"] == task:
#             if dp["Balance"] == "Linear":
#                 linear_means.append(dp["End Task Score"])
#             if dp["Balance"] == "Balanced":
#                 balanced_means.append(dp["End Task Score"])
#             ttype = dp["Task Type"]
#     print(task, np.mean(linear_means), np.mean(balanced_means))
#     differences.append({
#         "Percentage Improvement of Linear-Unbalanced vs. Balanced": (np.mean(linear_means) - np.mean(balanced_means)) / np.mean(balanced_means),
#         "Task": task,
#         "Task Type": ttype}
#     )
#
# df = pandas.DataFrame(datapoints)
#
# sns.set(font_scale=10)
# sns.color_palette("bright")
# sns.set_theme(style="whitegrid", font_scale=1.8)
#
#
# plt.figure(figsize=(8, 8))
# plt.title("Percentage Improvement of Encoders Trained on\nLinear-Unbalanced ImageNet vs. Encoders Trained on Balanced ImageNet")
# ax = sns.barplot(x="Percentage Improvement of Linear-Unbalanced vs. Balanced", y="Task", hue="Task Type", dodge=False, palette="pastel", data=pandas.DataFrame(differences))
# ax.legend(loc='lower left')
# plt.savefig("graphs/IN_balance/linear_vs_balanced.pdf", bbox_inches='tight')
#
#


# log = [dp["End Task Score"] for dp in datapoints if dp["Balance"] == "Log"]
# linear = [dp["End Task Score"] for dp in datapoints if dp["Balance"] == "Linear"]
# balanced = [dp["End Task Score"] for dp in datapoints if dp["Balance"] == "Balanced"]
# print()
#
# x = scipy.stats.kstest(log, balanced, N=len(log))
# print(x)

# for task in REAL_NAMES.values():
#     log = [dp["End Task Score"] for dp in datapoints if dp["Balance"] == "Log" and dp["Task"] == task]
#     linear = [dp["End Task Score"] for dp in datapoints if dp["Balance"] == "Linear" and dp["Task"] == task]
#     balanced = [dp["End Task Score"] for dp in datapoints if dp["Balance"] == "Balanced" and dp["Task"] == task]
#     F, p = scipy.stats.f_oneway(log, linear, balanced)
#     print(task, F, p)


############ Table 1.

# make_csv()
# data = pandas.read_csv("results.csv")
# # data = data[data["Method"] == "SWAV"]
# data = data[data["Epochs"] == 200]
# # data = data[data["Updates"] == int(1e6 * 50)]
# # data = data[data["Epochs"] == 200]
# data = data.set_index("Encoder")
#
# l1_log = []
# miou_log = []
# acc_log = []
#
# l1_balanced = []
# miou_balanced  = []
# acc_balanced  = []
#
# for task in ALL_TASKS:
#     means = []
#     for encoder in data.index:
#         if not math.isnan(data.loc[encoder][task]):
#             means.append(data.loc[encoder][task])
#     mean = np.mean(means)
#     for encoder in data.index:
#         if not math.isnan(data.loc[encoder][task]):
#             if data.loc[encoder]["Balance"] == "Log":
#                 if task in L1_TASKS:
#                     l1_log.append(data.loc[encoder][task])
#                 if task in MIOU_TASKS:
#                     miou_log.append(data.loc[encoder][task])
#                 if task in ACC_TASKS:
#                     acc_log.append(data.loc[encoder][task])
#             if data.loc[encoder]["Balance"] == "Balanced":
#                 if task in L1_TASKS:
#                     l1_balanced.append(data.loc[encoder][task])
#                 if task in MIOU_TASKS:
#                     miou_balanced.append(data.loc[encoder][task])
#                 if task in ACC_TASKS:
#                     acc_balanced.append(data.loc[encoder][task])
#
# print("Log vs Balanced")
# print("Accuracy:", scipy.stats.f_oneway(acc_log, acc_balanced ))
# print("mIOU:", scipy.stats.f_oneway(miou_log, miou_balanced ))
# print("L1", scipy.stats.f_oneway(l1_log, l1_balanced))
#
#
# make_csv()
# data = pandas.read_csv("results.csv")
# data = data[data["Updates"] == int(1e6 * 100)]
# data = data.set_index("Encoder")
#
# l1_linear = []
# miou_linear = []
# acc_linear = []
#
# l1_balanced = []
# miou_balanced  = []
# acc_balanced  = []
#
# for task in ALL_TASKS:
#     means = []
#     for encoder in data.index:
#         if not math.isnan(data.loc[encoder][task]):
#             means.append(data.loc[encoder][task])
#     mean = np.mean(means)
#     for encoder in data.index:
#         if not math.isnan(data.loc[encoder][task]):
#             if data.loc[encoder]["Balance"] == "Linear":
#                 if task in L1_TASKS:
#                     l1_linear.append(data.loc[encoder][task])
#                 if task in MIOU_TASKS:
#                     miou_linear.append(data.loc[encoder][task])
#                 if task in ACC_TASKS:
#                     acc_linear.append(data.loc[encoder][task])
#             if data.loc[encoder]["Balance"] == "Balanced":
#                 if task in L1_TASKS:
#                     l1_balanced.append(data.loc[encoder][task])
#                 if task in MIOU_TASKS:
#                     miou_balanced.append(data.loc[encoder][task])
#                 if task in ACC_TASKS:
#                     acc_balanced.append(data.loc[encoder][task])
#
# print("Linear vs Balanced")
# print("Accuracy:", scipy.stats.f_oneway(acc_linear, acc_balanced ))
# print("mIOU:", scipy.stats.f_oneway(miou_linear, miou_balanced ))
# print("L1", scipy.stats.f_oneway(l1_linear, l1_balanced))
#
#
# make_csv()
# data = pandas.read_csv("results.csv")
# # data = data[data["Method"] == "SWAV"]
# # data = data[data["Epochs"] == 200]
# data = data[(data["Updates"] == int(1e6 * 50)) | (data["Updates"] == int(1e6 * 100))]
# data = data.set_index("Encoder")
#
# datapoints = []
# for model in data.index:
#     for task in ALL_TASKS:
#         if not math.isnan(data.loc[model][task]):
#             datapoints.append({
#                 "Method": data.loc[model]["Method"],
#                 "Balance": data.loc[model]["Balance"],
#                 "Updates": data.loc[model]["Updates"],
#                 "Dataset Size": data.loc[model]["DatasetSize"],
#                 "Task": REAL_NAMES[task],
#                 "Score": data.loc[model][task]
#             })
# df = pandas.DataFrame(datapoints)
# df.to_csv("log_vs_balanced.csv")
#
# make_csv()
# data = pandas.read_csv("results.csv")
# # data = data[data["Method"] == "SWAV"]
# data = data[data["Epochs"] == 200]
# data = data[data["Updates"] == int(1e6 * 100)]
# data = data.set_index("Encoder")
#
# datapoints = []
# for model in data.index:
#     for task in ALL_TASKS:
#         datapoints.append({
#             "Method": data.loc[model]["Method"],
#             "Balance": data.loc[model]["Balance"],
#             "Task": REAL_NAMES[task],
#             "Score": data.loc[model][task]
#         })
# df = pandas.DataFrame(datapoints)
# df.to_csv("linear_vs_balanced.csv")
