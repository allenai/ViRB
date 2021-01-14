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
    for name, number in res:
        sn = name.replace("Imagenet", "IN")
        experiment_results[sn][task] = number
        experiment_results[sn][task+"-rank"] = rankings.index(name)+1

with open('results.csv', mode='w') as csv_file:
    fieldnames = ["Encoder", "Method"] + ALL_TASKS + [task+"-rank" for task in ALL_TASKS]
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
        row = {"Encoder": name, "Method": method}
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
# sns.set_theme()
# plt.figure(figsize=(20, 10))
#
data = pandas.read_csv("results.csv")
# results = data.sort_values("Imagenet", ascending=False).reset_index()
# g = sns.barplot(x="Imagenet", y="Encoder", hue="Method", data=results, dodge=False)
# for _, data in results.iterrows():
#     g.text(data.Imagenet - 0.015, data.name + 0.12, round(data.Imagenet, 4), color='white', ha="center", size=10, weight='bold')
# plt.savefig("imagenet-plot.png", dpi=100)
# plt.show()

#### Generating Pearson and Spearman Correlations
n = len(ALL_TASKS)
spearman = np.zeros((n,n))
pearson = np.zeros((n,n))
spearman_pval = np.zeros((n,n))
pearson_pval = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        values_i = data[ALL_TASKS[i]]
        values_j = data[ALL_TASKS[j]]
        s, sp = scipy.stats.spearmanr(values_i, values_j)
        p, pp = scipy.stats.pearsonr(values_i, values_j)
        spearman[i][j] = s
        pearson[i][j] = p
        spearman_pval[i][j] = sp
        pearson_pval[i][j] = pp
ax = sns.heatmap(spearman, annot=True)
ax.set_yticklabels(ALL_TASKS, rotation=0)
ax.set_xticklabels(ALL_TASKS, rotation=30, rotation_mode="anchor", ha='right', va="center")
plt.show()
