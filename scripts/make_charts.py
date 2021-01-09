import json
import glob
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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
    'SWAV_200',
    'SWAVCombination',
    'SWAVTaskonomy',
    'SWAVKinetics',
    'SWAVPlaces',
    'SWAVHalfImagenet',
    'SWAVLogImagenet'
]

ALL_TASKS = [
    "Pets",
    "SUN397",
    "CIFAR-100",
    "CalTech-101",
    "Eurosat",
    "dtd",
    "CLEVERNumObjects",
    "Imagenet"
]

def get_best_result(experiments, run, include_names=False):
    res = []
    for e in experiments:
        datapoints = []
        datapoint_files = glob.glob("out/%s/%s*/results.json" % (e, run))
        if len(datapoint_files) == 0:
            continue
        for f in datapoint_files:
            with open(f) as fp:
                datapoints.append(float(json.load(fp)["test_accuracy"]))
        if include_names:
            res.append((e, max(datapoints)))
        else:
            res.append(max(datapoints))
    return res


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
experiment_results = {name: {} for name in ALL_EXPERIMENTS}
for task in ["Imagenet"]:
    res = get_best_result(ALL_EXPERIMENTS, task, include_names=True)
    for name, number in res:
        experiment_results[name][task] = number
with open('Imagenet-results.csv', mode='w') as csv_file:
    fieldnames = ["Encoder"] + ALL_TASKS
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for name, results in experiment_results.items():
        row = {"Encoder": name}
        row.update(results)
        writer.writerow(row)


#### Generating Pearson and Spearman Correlations
# n = len(datasets)
# spearman = np.zeros((n,n))
# pearson = np.zeros((n,n))
# spearman_pval = np.zeros((n,n))
# pearson_pval = np.zeros((n,n))
# for i in range(n):
#     for j in range(n):
#         models_to_use = [model for model in data[datasets[i]] if model in data[datasets[j]]]
#         values_i = [data[datasets[i]][model] for model in models_to_use]
#         values_j = [data[datasets[j]][model] for model in models_to_use]
#         s, sp = scipy.stats.spearmanr(values_i, values_j)
#         p, pp = scipy.stats.pearsonr(values_i, values_j)
#         spearman[i][j] = s
#         pearson[i][j] = p
#         spearman_pval[i][j] = sp
#         pearson_pval[i][j] = pp

