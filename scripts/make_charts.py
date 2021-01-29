import json
import glob
import csv
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas

import scipy
import scipy.stats
import os


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
    'SWAV_50'
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

EMBEDDING_SEMANTIC_TASKS = [
    "Imagenet",
    "Pets",
    "CIFAR-100",
    "CalTech-101",
    "Eurosat",
    "dtd",
    "SUN397",
]
EMBEDDING_STRUCTURAL_TASKS = [
    "CLEVERNumObjects",
    "THORNumSteps",
    "THORActionPrediction"
]
PIXELWISE_SEMANTIC_TASKS = [
    "PetsDetection",
]
PIXELWISE_STRUCTURAL_TASKS = [
    "NYUDepth",
    "NYUWalkable",
    "THORDepth",
]

ALL_TASKS = EMBEDDING_SEMANTIC_TASKS + EMBEDDING_STRUCTURAL_TASKS + PIXELWISE_SEMANTIC_TASKS + PIXELWISE_STRUCTURAL_TASKS
EMBEDDING_TASKS = EMBEDDING_SEMANTIC_TASKS + EMBEDDING_STRUCTURAL_TASKS
PIXEL_TASKS = PIXELWISE_STRUCTURAL_TASKS + PIXELWISE_SEMANTIC_TASKS

REVERSED_SUCCESS_TASKS = [
    "TaskonomyInpainting",
    "TaskonomyEdges",
    "NYUDepth",
    "THORDepth",
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
experiment_results = {name.replace("Imagenet", "IN"): {} for name in ALL_EXPERIMENTS}
for task in ALL_TASKS:
    if task in REVERSED_SUCCESS_TASKS:
        res = get_best_result(ALL_EXPERIMENTS, task, include_names=True, c=-1.0)
    else:
        res = get_best_result(ALL_EXPERIMENTS, task, include_names=True)
    rankings, _ = zip(*sorted(res, key=lambda x: x[1], reverse=True))
    mean = [r for n, r in res if n == "Supervised"][0]
    std = np.std([r for _, r in res])
    for name, number in res:
        sn = name.replace("Imagenet", "IN")
        experiment_results[sn][task] = number
        experiment_results[sn][task+"-rank"] = rankings.index(name)+1
        experiment_results[sn][task+"-normalized"] = (number - mean) / std

with open('results.csv', mode='w') as csv_file:
    fieldnames = ["Encoder", "Method", "Dataset", "Epochs", "Updates"] + ALL_TASKS + \
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

        if "_1000" in name:
            epochs = 1000
        elif "_800" in name:
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
            "Epochs": epochs,
            "Updates": epochs * dataset_size
        }
        row.update(results)
        writer.writerow(row)

### BIG TABLE
# res = get_best_result(ALL_EXPERIMENTS, "Imagenet", include_names=True)
# res.sort(key=lambda x: x[1])
# names = []
# results = []
# labels = []
# for name, value in res:
#     names.append(name.replace("Imagenet", "IN"))
#     results.append(round(value, 4))
#     if "MoCo" in name:
#         label = "MoCo"
#     elif "SWAV" in name:
#         label = "SWAV"
#     elif "PIRL" in name:
#         label = "PIRL"
#     elif "SimCLR" in name:
#         label = "SimCLR"
#     elif "Supervised" in name:
#         label = "Supervised"
#     elif "Random" in name:
#         label = "Random"
#     else:
#         label = "Other"
#     labels.append(label)
# make_ranked_bar_chart(names, results, "Top-1 Accuracy", "Imagenet Classification", labels=labels)
#
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
# sns.set_theme()
# colors = sns.color_palette()
# palette = {method: colors[i] for i, method in enumerate(set(data["Method"]))}
# for task in ALL_TASKS:
#     plt.figure(figsize=(20, 10))
#     data = pandas.read_csv("results.csv")
#     data = data.set_index("Encoder")
#     for t in ALL_TASKS:
#         # rand = data.loc['Random', t]
#         # for encoder in data.index:
#         #     data.loc[encoder, t] = data.loc[encoder, t] - rand
#         sup = data.loc['Supervised', t]
#         for encoder in data.index:
#             data.loc[encoder, t] = data.loc[encoder, t] - sup
#     #data = data.drop("Random")
#     data = data.drop("SimCLR_1000")
#     results = data.reset_index()
#     results = results.sort_values(task, ascending=False).reset_index()
#     g = sns.barplot(x=task, y="Encoder", hue="Method", data=results, dodge=False, palette=palette)
#     sign = 1.0 if results[task][1] >= 0 else -1.0
#     for _, data in results.iterrows():
#         g.text(data[task] - (sign * 0.08), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
#     plt.title("%s Test Results" % task)
#     plt.xlabel("Test Performance")
#     #plt.show()
#     plt.savefig("new_graphs/%s-test-results-subtracted.png" % task, dpi=100)
#     plt.clf()

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
# data = pandas.read_csv("results.csv")
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
# title = "Spearman Correlation on Performance Between Tasks IN POV"
# plt.title(title)
# ax = sns.heatmap(spearman, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+"-1.png")
# plt.clf()
# title = "Spearman Correlation p-values on Performance Between Tasks IN POV"
# plt.title(title)
# ax = sns.heatmap(spearman_pval, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
# plt.clf()
# title = "Pearson Correlation on Performance Between Tasks IN POV"
# plt.title(title)
# ax = sns.heatmap(pearson, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
# plt.clf()
# title = "Pearson Correlation p-values on Performance Between Tasks IN POV"
# plt.title(title)
# ax = sns.heatmap(pearson_pval, annot=True)
# ax.set_yticklabels(tasks, rotation=0)
# ax.set_xticklabels(tasks, rotation=30, rotation_mode="anchor", ha='right', va="center")
# plt.savefig("graphs/"+title.replace(" ", "_")+".png")
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


##### Make Per Task Vategory Violin Plot of Scores for MoCo and SWAV 200
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
# plt.title("SWAV and MoCov2 200 Modles, 1.3M Image Datasets Performance")
# df = pandas.DataFrame(per_task_ms_200_full)
# sns.violinplot(x="Task", y="NormalizedScore", hue="TaskType", data=df)
# plt.savefig("graphs/mocov2_and_swav_200_different_datasets/violin.png", dpi=100)
# plt.clf()

##### Make Per Task Vategory Violin Plot of Scores for MoCo 200
# data = pandas.read_csv("results.csv")
# ms_data = data[(data["Method"] == "MoCo")]
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
# plt.title("MoCov2 200 Models, 1.3M Image Datasets Per Category Performance")
# df = pandas.DataFrame(per_task_ms_200_full)
# sns.violinplot(x="Task", y="NormalizedScore", hue="TaskType", data=df)
# plt.savefig("graphs/mocov2_200_different_datasets/violin.png", dpi=100)
# plt.clf()
#
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

data = pandas.read_csv("results.csv")
sns.set_theme()
colors = sns.color_palette()
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
#     plt.savefig("graphs/in_subsets/%s_vs_Supervised.png" % dataset, dpi=100)
#     plt.clf()


d = data[(data["Updates"] == int(100*1e6)) & (data["Method"] == "MoCo")]
plt.figure(figsize=(20, 10))
ee = d.set_index("Encoder")
vals = []
for task in [t for t in ALL_TASKS if t not in REVERSED_SUCCESS_TASKS]:
    for encoder in d["Encoder"]:
        if encoder == "SWAVKinetics" and task == "THORDepth":
            continue

        if task in EMBEDDING_SEMANTIC_TASKS:
            ttype = "Embedding-Semantic"
        elif task in EMBEDDING_STRUCTURAL_TASKS:
            ttype = "Embedding-Structural"
        elif task in PIXELWISE_SEMANTIC_TASKS:
            ttype = "Pixelwise-Semantic"
        elif task in PIXELWISE_STRUCTURAL_TASKS:
            ttype = "Pixelwise-Structural"
        else:
            ttype = "Other"

        vals.append({
            "Encoder": encoder,
            "Task": task,
            "Method": ee.loc[encoder, "Method"],
            "Dataset": ee.loc[encoder, "Dataset"],
            "Score": ee.loc[encoder, task],
            "SupervisedScore": data.set_index("Encoder").loc["Supervised", task],
            "TaskType": ttype
        })

plt.title("Half Imagenet Enocders Performance and Equivalent vs. Supervised")
df = pandas.DataFrame(vals)
g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Dataset", style="Method", data=df, s=100)
sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["Supervised IN Results", "Supervised IN Results"], palette={"IN Results": "Red"})
plt.xlabel("Supervised Performance")
plt.ylabel("Self Supervised Encoder Model Performance")
plt.savefig("graphs/Imagenet_Subsets/HalfEquivalent_vs_Supervised.png", dpi=100)
plt.clf()

d = data[(data["Updates"] == int(100*1e6))]
plt.figure(figsize=(20, 10))
ee = d.set_index("Encoder")
vals = []
for task in [t for t in ALL_TASKS if t not in REVERSED_SUCCESS_TASKS]:
    for encoder in d["Encoder"]:
        if encoder == "SWAVKinetics" and task == "THORDepth":
            continue

        if task in EMBEDDING_SEMANTIC_TASKS:
            ttype = "Embedding-Semantic"
        elif task in EMBEDDING_STRUCTURAL_TASKS:
            ttype = "Embedding-Structural"
        elif task in PIXELWISE_SEMANTIC_TASKS:
            ttype = "Pixelwise-Semantic"
        elif task in PIXELWISE_STRUCTURAL_TASKS:
            ttype = "Pixelwise-Structural"
        else:
            ttype = "Other"

        vals.append({
            "Encoder": encoder,
            "Task": task,
            "Method": ee.loc[encoder, "Method"],
            "Dataset": ee.loc[encoder, "Dataset"],
            "Score": ee.loc[encoder, task],
            "SupervisedScore": data.set_index("Encoder").loc["Supervised", task],
            "TaskType": ttype
        })

plt.title("Half Imagenet Enocders Performance and Equivalent vs. Supervised")
df = pandas.DataFrame(vals)
g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Dataset", style="Method", data=df, s=100)
sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["Supervised IN Results", "Supervised IN Results"], palette={"IN Results": "Red"})
plt.xlabel("Supervised Performance")
plt.ylabel("Self Supervised Encoder Model Performance")
plt.savefig("graphs/Imagenet_Subsets/HalfEquivalent_vs_Supervised.png", dpi=100)
plt.clf()

d = data[(data["Updates"] == int(50*1e6))]
plt.figure(figsize=(20, 10))
ee = d.set_index("Encoder")
vals = []
for task in [t for t in ALL_TASKS if t not in REVERSED_SUCCESS_TASKS]:
    for encoder in d["Encoder"]:
        if encoder == "SWAVKinetics" and task == "THORDepth":
            continue

        if task in EMBEDDING_SEMANTIC_TASKS:
            ttype = "Embedding-Semantic"
        elif task in EMBEDDING_STRUCTURAL_TASKS:
            ttype = "Embedding-Structural"
        elif task in PIXELWISE_SEMANTIC_TASKS:
            ttype = "Pixelwise-Semantic"
        elif task in PIXELWISE_STRUCTURAL_TASKS:
            ttype = "Pixelwise-Structural"
        else:
            ttype = "Other"

        vals.append({
            "Encoder": encoder,
            "Task": task,
            "Method": ee.loc[encoder, "Method"],
            "Dataset": ee.loc[encoder, "Dataset"],
            "Score": ee.loc[encoder, task],
            "SupervisedScore": data.set_index("Encoder").loc["Supervised", task],
            "TaskType": ttype
        })

plt.title("Quarter Imagenet Enocders Performance and Equivalent vs. Supervised")
df = pandas.DataFrame(vals)
g = sns.scatterplot(x="SupervisedScore", y="Score", hue="Dataset", style="Method", data=df, s=100)
sns.lineplot(x=[0.5,1], y=[0.5,1], ax=g, hue=["Supervised IN Results", "Supervised IN Results"], palette={"IN Results": "Red"})
plt.xlabel("Supervised Performance")
plt.ylabel("Self Supervised Encoder Model Performance")
plt.savefig("graphs/Imagenet_Subsets/QuarterEquivalent_vs_Supervised.png", dpi=100)
plt.clf()