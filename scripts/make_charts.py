import json
import glob
import matplotlib.pyplot as plt


def get_best_result(experiments, run):
    res = []
    for e in experiments:
        datapoints = []
        datapoint_filels = glob.glob("out/%s/%s*/results.json" % (e, run))
        for f in datapoint_filels:
            with open(f) as fp:
                datapoints.append(float(json.load(fp)["test_accuracy"]))
        res.append((e, max(datapoints)))
    return res

print(get_best_result(["Supervised-end-to-end", "Supervised-full-channel"], "TaskonomyInpainting"))
