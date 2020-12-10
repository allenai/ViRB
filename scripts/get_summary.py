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
        training_runs = glob.glob(experiment + "/" + task + "*")
        if len(training_runs) > 0:
            results[experiment_name][task] = {"training_runs": {}}
        for training_run in (training_runs):
            with open(training_run+"/results.json") as f:
                run_results = json.load(f)
            results[experiment_name][task]["training_runs"][training_run.replace(task+"-", "")] = run_results
        if task in results[experiment_name]:
            best_test_config = None
            best_test_result = 0.0
            for name, training_run in results[experiment_name][task]["training_runs"].items():
                if training_run["test_accuracy"] > best_test_result:
                    best_test_result = training_run["test_accuracy"]
                    best_test_config = name
            results[experiment_name][task]["best_test_config"] = best_test_config
            results[experiment_name][task]["best_test_result"] = best_test_result

with open("../results.json") as f:
    json.dump(results, f)
