import json


def get_all_task_names(data):
    tasks = set()
    for name, experiment in data.items():
        for task in experiment:
            tasks.add(task)
    return list(tasks)


def get_ranking_for_task(data, task):
    rankings = []
    for name, experiment in data.items():
        if task in experiment:
            rankings.append((name, experiment[task]["best_test_config"], experiment[task]["best_test_result"]))
    rankings.sort(key=lambda x: x[1])
    return rankings


rankings = {}
with open("../results.json") as f:
    results = json.load(f)
tasks = get_all_task_names(results)
for task in tasks:
    rankings[task] = get_ranking_for_task(results, task)
print(rankings)

with open("../rankings.json", 'w') as out:
    json.dump(rankings, out)
