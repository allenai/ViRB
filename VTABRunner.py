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
    "CLEVERDist",
    "SUN397"
]
BINARY_PIXEL_WISE_CLASSIFICATION = [
    "Flowers-Detection",
    "Pets-Detection"
]
PIXEL_WISE_REGRESSION = [
    "THORDepth"
]

GPU_IDS = ["cuda:%d" % i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() > 0 else ["cpu"]

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
    if config["task"] == "Pets-Detection":
        from datasets.PetsDetectionEncodbleDataset import PetsDetectionEncodableDataset
        return PetsDetectionEncodableDataset
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
    if config["task"] == "SUN397":
        from datasets.SUN397EncodbleDataset import SUN397EncodableDataset
        return SUN397EncodableDataset
    if config["task"] == "THORDepth":
        from datasets.ThorDepthEncodbleDataset import ThorDepthEncodableDataset
        return ThorDepthEncodableDataset


def get_task_head(config, dataset):
    if config["task"] in PIXEL_WISE_REGRESSION:
        from models.PixelWisePredictionHead import PixelWisePredictionHead
        return PixelWisePredictionHead(1)
    if config["task"] in BINARY_PIXEL_WISE_CLASSIFICATION:
        from models.PixelWisePredictionHead import PixelWisePredictionHead
        return PixelWisePredictionHead(1)
    if config["task"] in CLASSIFICATION_TASKS:
        if "embedding" not in config["encoder"].outputs():
            raise Exception("A model needs to have an embedding output in order to be tested on classification tasks!")
        from models.ClassificationHead import ClassificationHead
        return ClassificationHead(config["encoder"].outputs()["embedding"][0], dataset.num_classes())


def get_optimizer(config, model):
    if config["optimizer"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])
    if config["optimizer"].lower() == "sgd":
        if "momentum" in config:
            return torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
        else:
            return torch.optim.SGD(model.parameters(), lr=config["lr"])


def get_scheduler(config, optimizer):
    if "scheduler" not in config:
        return None
    if config["scheduler"]["type"] == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            config["scheduler"]["type"]["step_size"],
            gamma=config["scheduler"]["type"]["gamma"]
        )


def get_loss_function(config):
    if config["task"] in PIXEL_WISE_REGRESSION:
        return torch.nn.SmoothL1Loss()
    if config["task"] in CLASSIFICATION_TASKS:
        return torch.nn.CrossEntropyLoss()
    if config["task"] in BINARY_PIXEL_WISE_CLASSIFICATION:
        return torch.nn.BCEWithLogitsLoss()


def get_error_function(config):
    if config["task"] in PIXEL_WISE_REGRESSION:
        return torch.nn.SmoothL1Loss()
    if config["task"] in CLASSIFICATION_TASKS:
        from utils.error_functions import classification_error
        return classification_error
    if config["task"] in BINARY_PIXEL_WISE_CLASSIFICATION:
        from utils.error_functions import binary_pixel_wise_prediction_loss
        return binary_pixel_wise_prediction_loss


def run_VTAB_task(config):

    cpu_name = mp.current_process().name
    cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
    gpu_id = GPU_IDS[cpu_id]
    print("CPU ID %d, GPU ID %s" % (cpu_id, gpu_id))

    dataset_class = get_dataset_class(config)
    trainset = dataset_class(train=True)
    testset = dataset_class(train=False)
    loss_function = get_loss_function(config)
    error_function = get_error_function(config)

    training_configs = []
    print("AAA")
    for tc_name, tc in config["training_configs"].items():
        encoder = config["encoder"]
        task_head = get_task_head(config, trainset)
        model = VTABModel(encoder, task_head, train_encoder=config["train_encoder"])
        optimizer = get_optimizer(tc, model)
        scheduler = get_scheduler(tc, optimizer)
        training_configs.append({
            "name": tc_name,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler
        })
    print("BBB")
    pre_encode = config["pre_encode"] if "pre_encode" in config else None
    task = VTABTask(
        config["experiment_name"],
        config["task_name"],
        training_configs=training_configs,
        train_set=trainset,
        test_set=testset,
        loss=loss_function,
        error=error_function,
        out_dir="out/"+config["experiment_name"]+"/"+config["task_name"],
        num_workers=config["num_workers"],
        device=gpu_id,
        pre_encode=pre_encode
    )
    results = task.run(config["num_epochs"])
    print("CCC")
    return results


class VTABRunner:

    def __init__(
            self,
            experiments,
            train_encoder=False,
            experiment_config_path="configs/vtab_configs/default.yaml",
            num_gpus=torch.cuda.device_count(),
            total_num_workers=12
    ):
        self.num_threads = num_gpus if num_gpus > 0 else 1
        with open(experiment_config_path) as file:
            tasks = yaml.load(file, Loader=yaml.FullLoader)
        self.experiment_queue = []
        for experiment_name, experiment_encoder in experiments.items():
            for task_name, task in tasks.items():
                experiment = copy.deepcopy(task)
                experiment["task_name"] = task_name
                experiment["experiment_name"] = experiment_name
                experiment["encoder"] = experiment_encoder
                experiment["train_encoder"] = train_encoder
                experiment["num_workers"] = total_num_workers // self.num_threads
                self.experiment_queue.append(experiment)

    def run(self):
        pool = mp.Pool(min(len(GPU_IDS), len(self.experiment_queue)))
        pool.map(run_VTAB_task, self.experiment_queue)

        # if self.num_threads == 1 or len(self.experiment_queue) == 1:
        #     run_VTAB_queue(self.experiment_queue)
        # else:
        #     experiments_per_device = [[] for _ in range(self.num_threads)]
        #     for experiment in self.experiment_queue:
        #         idx = int(experiment["device"][-1])
        #         experiments_per_device[idx].append(experiment)
        #     mp.freeze_support()
        #     mp.set_start_method('spawn')
        #     procs = [
        #         mp.Process(target=run_VTAB_queue, args=(experiments_per_device[i]))
        #         for i in range(self.num_threads)
        #     ]
        #     [proc.start() for proc in procs]
        #     [proc.join() for proc in procs]
