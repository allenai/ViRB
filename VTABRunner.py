import torch
import os
import yaml
import json
import copy
import multiprocessing as mp

from models.ResNet50Encoder import ResNet50Encoder
from models.VTABModel import VTABModel
from models.ClassificationHead import ClassificationHead
from tasks.VTABTask import VTABTask


CLASSIFICATION_TASKS = [
    "CalTech-101",
    "CIFAR-100",
    "Pets",
    "Eurosat",
    "dtd",
    "CLEVERNumObjects",
    "CLEVERDist"
]


def get_dataset_class(config):
    if config["task"] == "CalTech-101":
        from datasets.Caltech101EncldableDataset import CalTech101EncodableDataset
        return CalTech101EncodableDataset
    if config["task"] == "CIFAR-100":
        from datasets.Cifar100EncodbleDataset import CIFAR100EncodableDataset
        return CIFAR100EncodableDataset
    if config["task"] == "Pets":
        from datasets.PetsEncodbleDataset import PetsEncodableDataset
        return PetsEncodableDataset
    if config["task"] == "Eurosat":
        from datasets.EurosatEncodbleDataset import EurosatEncodableDataset
        return EurosatEncodableDataset
    if config["task"] == "dtd":
        from datasets.dtdEncodbleDataset import dtdEncodableDataset
        return dtdEncodableDataset
    if config["task"] == "CLEVERNumObjects":
        from datasets.CLEVERNumObjectsEncodbleDataset import CLEVERNumObjectsEncodableDataset
        return CLEVERNumObjectsEncodableDataset
    if config["task"] == "CLEVERDist":
        from datasets.CLEVERDistEncodbleDataset import CLEVERDistEncodableDataset
        return CLEVERDistEncodableDataset


def get_task_head(config, dataset):
    if config["task"] in CLASSIFICATION_TASKS:
        if "embedding" not in config["output_shape"]:
            raise Exception("A model needs to have an embedding output in order to be tested on classification tasks!")
        from models.ClassificationHead import ClassificationHead
        return ClassificationHead(config["output_shape"]["embedding"][0], dataset.num_classes())


def get_optimizer(config, model):
    if config["optimizer"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])
    if config["optimizer"].lower() == "sgd":
        if "momentum" in config:
            return torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
        else:
            return torch.optim.SGD(model.parameters(), lr=config["lr"])


def get_loss_function(config):
    if config["task"] in CLASSIFICATION_TASKS:
        return torch.nn.CrossEntropyLoss()


def get_error_function(config):
    if config["task"] in CLASSIFICATION_TASKS:
        from utils.error_functions import classification_error
        return classification_error


def run_VTAB_task(config):
    dataset_class = get_dataset_class(config)
    trainset = dataset_class(train=True)
    testset = dataset_class(train=False)
    encoder = copy.deepcopy(config["encoder"])
    task_head = get_task_head(config, trainset)
    model = VTABModel(encoder, task_head, train_encoder=config["train_encoder"])
    loss_function = get_loss_function(config)
    error_function = get_error_function(config)
    optimizer = get_optimizer(config, model)
    task = VTABTask(
        name=config["name"],
        task=config["task"],
        model=model,
        train_set=trainset,
        test_set=testset,
        loss=loss_function,
        error=error_function,
        optimizer=optimizer,
        out_dir="out/"+config["run_name"]+"/"+config["name"],
        num_workers=config["num_workers"],
        device=config["device"]
    )
    results = task.run(config["num_epochs"])
    return results


def run_VTAB_queue(queue, return_dict=None):
    if return_dict is None:
        return_dict = {}
    for experiment in queue:
        return_dict[experiment["name"]] = run_VTAB_task(experiment)
    return return_dict


class VTABRunner:

    def __init__(
            self,
            encoder,
            run_name,
            output_shape={"embedding": torch.Size([2048])},
            train_encoder=False,
            experiment_config_path="configs/default.yaml",
            num_gpus=torch.cuda.device_count(),
            total_num_workers=12
    ):
        self.num_threads = num_gpus if num_gpus > 0 else 1
        with open(experiment_config_path) as file:
            experiments = yaml.load(file, Loader=yaml.FullLoader)
        self.experiment_queue = []
        for i, (name, experiment) in enumerate(experiments.items()):
            experiment["name"] = name
            experiment["run_name"] = run_name
            experiment["encoder"] = encoder
            experiment["train_encoder"] = train_encoder
            experiment["output_shape"] = output_shape
            experiment["device"] = "cuda:%d" % (i % num_gpus) if num_gpus > 0 else "cpu"
            experiment["num_workers"] = total_num_workers // self.num_threads
            self.experiment_queue.append(experiment)

    def run(self):
        if self.num_threads == 1:
            results = run_VTAB_queue(self.experiment_queue)
        else:
            experiments_per_device = [[] for _ in range(self.num_threads)]
            for experiment in self.experiment_queue:
                idx = int(experiment["device"][-1])
                experiments_per_device[idx].append(experiment)
            mp.freeze_support()
            mp.set_start_method('spawn')
            manager = mp.Manager()
            results = manager.dict()
            procs = [
                mp.Process(target=run_VTAB_queue, args=(experiments_per_device[i], results))
                for i in range(self.num_threads)
            ]
            [proc.start() for proc in procs]
            [proc.join() for proc in procs]
        with open("out/results.json", "r+") as f:
            if os.exists("out/results.json"):
                current_results = json.load(f)
            else:
                current_results = {}
            new_results = current_results + results
            json.dump(new_results)
