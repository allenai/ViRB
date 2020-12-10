import glob
import json

TASKS = [
    "CalTech-101",
    "CIFAR-100",
    "Pets",
    "Eurosat",
    "dtd",
    "CLEVERNumObjects",
    "CLEVERDist",
    "SUN397"
    "Flowers-Detection",
    "Pets-Detection"
    "THORDepth"
]

results = {}
for experiment in glob.glob("../out/*"):
    experiment_name = experiment.split("/")[-1]
    results[experiment_name] = {}
    for task in TASKS:
        training_runs = experiment + "/" + task + "*"
        if len(training_runs) > 0:
            results[experiment_name][task] = {}
        for training_run in (training_runs):
            with open(training_run) as f:
                run_results = json.load(f)
            results[experiment_name][task][training_run.replace(task+"-", "")] = run_results

with open("../results.json") as f:
    json.dump(results, f)
