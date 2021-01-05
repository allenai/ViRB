import json
import glob
import matplotlib.pyplot as plt


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

print(get_best_result(["Random-full-channel", "Supervised-full-channel"], "TaskonomyInpainting"))
