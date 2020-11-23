import torch
import sys
import yaml
import copy

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
        model=model,
        train_set=trainset,
        test_set=testset,
        loss=loss_function,
        error=error_function,
        optimizer=optimizer,
        out_dir="out/"+config["run_name"]+"/"+config["name"]
    )
    task.run(config["num_epochs"])


class VTABRunner:

    def __init__(
            self,
            encoder,
            run_name,
            output_shape={"embedding": torch.Size([2048])},
            train_encoder=False,
            experiment_config_path="configs/default.yaml",
            num_gpus=0
    ):
        self.num_workers = num_gpus if num_gpus > 0 else 1
        with open(experiment_config_path) as file:
            experiments = yaml.load(file, Loader=yaml.FullLoader)
        self.experiment_queue = []
        for name, experiment in experiments.items():
            experiment["name"] = name
            experiment["run_name"] = run_name
            experiment["encoder"] = encoder
            experiment["train_encoder"] = train_encoder
            experiment["output_shape"] = output_shape
            self.experiment_queue.append(experiment)

    def run(self):
        for experiment in self.experiment_queue:
            run_VTAB_task(experiment)
