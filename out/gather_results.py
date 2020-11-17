import json
import glob


results = {}
for result in glob.glob("*/*/results.json"):
    parts = result.split("/")
    training_run = parts[0]
    experiment = parts[1]
    with open(result) as f:
        data = json.load(f)
    if training_run not in results:
        results[training_run] = {}
    results[training_run][experiment] = data['test_accuracy']

with open('combined_results.json', 'w') as f:
    json.dump(results, f, indent=4)

