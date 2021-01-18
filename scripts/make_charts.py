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
    'SWAVQuarterImagenet'
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

QUARTER_IMAGENET_EXPERIMENTS = [
    'MoCov2QuarterImagenet',
    'SWAVQuarterImagenet'
]
LOG_IMAGENET_EXPERIMENTS = [
    'SWAVLogImagenet',
    'MoCov2LogImagenet',
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

ALL_TASKS = [
    "Pets",
    "SUN397",
    "CIFAR-100",
    "CalTech-101",
    "Eurosat",
    "dtd",
    "CLEVERNumObjects",
    "Imagenet",
    "Pets-Detection",
    "NYUDepth",
    "NYUWalkable",
    "THORDepth",
    # "TaskonomyInpainting",
    # "TaskonomyEdges"
]

EMBEDDING_TASKS = [
    "Pets",
    "SUN397",
    "CIFAR-100",
    "CalTech-101",
    "Eurosat",
    "dtd",
    "CLEVERNumObjects",
    "Imagenet",
]
PIXEL_TASKS = [
    "Pets-Detection",
    "NYUDepth",
    "NYUWalkable",
    "THORDepth",
    "TaskonomyInpainting",
    "TaskonomyEdges"
]

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
        datapoint_files = glob.glob("out/%s/%s*/results.json" % (e, run))
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

def autolabel(rects, ax):
    """
        Attach a text label above each bar in *rects*, displaying its height. Copied from:
        https://matplotlib.org/3.3.3/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def make_ranked_bar_chart(names, results, success_metric, task, labels=None):
    x = np.arange(len(names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    if labels is None:
        rects = [ax.bar(x, results, width)]
    else:
        label_table = {
            label: (
                [x[i] for i in range(len(x)) if labels[i] == label],
                [results[i] for i in range(len(x)) if labels[i] == label]
            ) for label in labels
        }
        rects = [ax.bar(x, results, width, label=label) for label, (x, results) in label_table.items()]

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(success_metric)
    ax.set_title('Test performance of encoders on %s' % task)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=-90)
    ax.legend()

    for r in rects:
        autolabel(r, ax)
    fig.tight_layout()

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
# experiment_results = {name.replace("Imagenet", "IN"): {} for name in ALL_EXPERIMENTS}
# for task in ALL_TASKS:
#     if task in REVERSED_SUCCESS_TASKS:
#         res = get_best_result(ALL_EXPERIMENTS, task, include_names=True, c=-1.0)
#     else:
#         res = get_best_result(ALL_EXPERIMENTS, task, include_names=True)
#     rankings, _ = zip(*sorted(res, key=lambda x: x[1], reverse=True))
#     for name, number in res:
#         sn = name.replace("Imagenet", "IN")
#         experiment_results[sn][task] = number
#         experiment_results[sn][task+"-rank"] = rankings.index(name)+1
#
# with open('results.csv', mode='w') as csv_file:
#     fieldnames = ["Encoder", "Method"] + ALL_TASKS + [task+"-rank" for task in ALL_TASKS]
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()
#     for name, results in experiment_results.items():
#         if "MoCo" in name:
#             method = "MoCo"
#         elif "SWAV" in name:
#             method = "SWAV"
#         elif "PIRL" in name:
#             method = "PIRL"
#         elif "SimCLR" in name:
#             method = "SimCLR"
#         elif "Supervised" in name:
#             method = "Supervised"
#         elif "Random" in name:
#             method = "Random"
#         else:
#             method = "Other"
#         row = {"Encoder": name, "Method": method}
#         row.update(results)
#         writer.writerow(row)


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
#
# data = pandas.read_csv("results.csv")
# sns.set_theme()
# colors = sns.color_palette()
# palette = {method: colors[i] for i, method in enumerate(set(data["Method"]))}
# for task in ALL_TASKS:
#     plt.figure(figsize=(20, 10))
#     data = pandas.read_csv("results.csv")
#     results = data.sort_values(task, ascending=False).reset_index()
#     g = sns.barplot(x=task, y="Encoder", hue="Method", data=results, dodge=False, palette=palette)
#     sign = 1.0 if results[task][0] > 0 else -1.0
#     for _, data in results.iterrows():
#         g.text(data[task] - (sign * 0.02), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
#     plt.title("%s Test Results" % task)
#     plt.xlabel("Test Performance")
#     plt.savefig("graphs/%s-test-results.png" % task, dpi=100)
#     plt.clf()

#### Generating Pearson and Spearman Correlations
# data = pandas.read_csv("results.csv")
# tasks = ["Imagenet", "CalTech-101", "Pets", "Pets-Detection", "dtd", "CIFAR-100", "SUN397", "Eurosat",
#          "CLEVERNumObjects", "THORDepth", "NYUDepth", "NYUWalkable"]
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
# # title = "Spearman Correlation p-values on Performance Between Tasks IN POV"
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
#
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
data = pandas.read_csv("results.csv")
# normalized_scores = get_normalized_summed_scores(data)
# data = pandas.DataFrame(normalized_scores)
data = data.set_index("Encoder")
values = []
for task in ALL_TASKS:
    mocos_vals = []
    swav_vals = []
    for encoder in ALL_EXPERIMENTS:
        encoder = encoder.replace("Imagenet", "IN")
        if "MoCov2" in encoder:
            mocos_vals.append(data[task][encoder])
        if "SWAV" in encoder:
            swav_vals.append(data[task][encoder])
    values.append({"Method": "MoCov2", "task": task, "score": np.mean(mocos_vals)})
    values.append({"Method": "SWAV", "task": task, "score": np.mean(swav_vals)})
values = pandas.DataFrame(values)
sns.set_theme()
# normalized_scores = get_normalized_summed_scores(data)
# data = pandas.DataFrame(normalized_scores)
plt.figure(figsize=(20, 10))
# results = data.sort_values(task, ascending=False).reset_index()
g = sns.barplot(x="score", y="task", hue="Method", data=values)
# sign = 1.0 if results[task][0] > 0 else -1.0
# for _, data in results.iterrows():
#     g.text(data[task] - (sign * 0.02), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
plt.title("MoCo vs SWAV Test Results" )
plt.xlabel("Test Performance")
plt.show()
# plt.savefig("graphs/%s-groupped-test-results.png" % task, dpi=100)
plt.clf()

# data = pandas.read_csv("results.csv")
# # normalized_scores = get_normalized_summed_scores(data)
# # data = pandas.DataFrame(normalized_scores)
# data = data.set_index("Encoder")
# values = []
# for task in ALL_TASKS:
#     half_vals = []
#     quarter_vals = []
#     unbalanced_vals = []
#     log_vals = []
#     for encoder in ALL_EXPERIMENTS:
#         encoder = encoder.replace("Imagenet", "IN")
#         # if "Half" in encoder:
#         #     half_vals.append(data[task][encoder])
#         if "Quarter" in encoder:
#             quarter_vals.append(data[task][encoder])
#         # if "Unbalanced" in encoder:
#         #     unbalanced_vals.append(data[task][encoder])
#         if "Log" in encoder:
#             log_vals.append(data[task][encoder])
#     # values.append({"Dataset": "Half", "task": task, "score": np.mean(half_vals)})
#     values.append({"Dataset": "Quarter", "task": task, "score": np.mean(quarter_vals)})
#     # values.append({"Dataset": "Unbalanced", "task": task, "score": np.mean(unbalanced_vals)})
#     values.append({"Dataset": "Log", "task": task, "score": np.mean(log_vals)})
# values = pandas.DataFrame(values)
# sns.set_theme()
# # normalized_scores = get_normalized_summed_scores(data)
# # data = pandas.DataFrame(normalized_scores)
# plt.figure(figsize=(20, 10))
# # results = data.sort_values(task, ascending=False).reset_index()
# g = sns.barplot(x="score", y="task", hue="Dataset", data=values)
# # sign = 1.0 if results[task][0] > 0 else -1.0
# # for _, data in results.iterrows():
# #     g.text(data[task] - (sign * 0.02), data.name + 0.12, round(data[task], 4), color='white', ha="center", size=10, weight='bold')
# plt.title("Test Results of Encoders Trained on Quarter and Log ImageNet" )
# plt.xlabel("Test Performance")
# plt.show()
# # plt.savefig("graphs/%s-groupped-test-results.png" % task, dpi=100)
# plt.clf()

#### Generating Pearson and Spearman Correlations for IN Tasks
# data = pandas.read_csv("results.csv")
# data = data.set_index("Encoder", drop=False)
# data = data.loc[PLACES_EXPERIMENTS + TASKONOMY_EXPERIMENTS + COMBO_EXPERIMENTS + KINETICS_EXPERIMENTS]
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



