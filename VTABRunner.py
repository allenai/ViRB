import torch
import os
import sys
import yaml
import json
import copy
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import threading
import queue
import time
import curses
import traceback
from datetime import datetime

from models.ResNet50Encoder import ResNet50Encoder
from models.VTABModel import VTABModel
from models.ClassificationHead import ClassificationHead
from tasks.VTABTask import VTABTask
from utils.progress_data_packets import ProgressDataPacket


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
    "THORDepth",
    "NYUDepth"
]

GPU_IDS = ["cuda:%d" % i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() > 0 else ["cpu"]
GPU_MAP = {}

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
    if config["task"] == "NYUDepth":
        from datasets.NyuDepthEncodbleDataset import NyuDepthEncodableDataset
        return NyuDepthEncodableDataset


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
        from utils.error_functions import iou
        return iou


def run_VTAB_task(config, logging_queue):
    dataset_class = get_dataset_class(config)
    trainset = dataset_class(train=True)
    testset = dataset_class(train=False)
    loss_function = get_loss_function(config)
    error_function = get_error_function(config)
    training_configs = []
    for tc_name, tc in config["training_configs"].items():
        encoder = copy.deepcopy(config["encoder"])
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
        logging_queue=logging_queue,
        num_workers=config["num_workers"],
        device=config["device_id"],
        pre_encode=pre_encode
    )
    results = task.run(config["num_epochs"])
    return results


def thread_loop(gpu_id, config_queue, logging_queue):
    while True:
        try:
            conf = config_queue.get(timeout=1)
            conf["device_id"] = gpu_id
            logging_queue.put(ProgressDataPacket(
                name="Initializing",
                device=gpu_id,
                new_task=False
            ))
            run_VTAB_task(conf, logging_queue)
            logging_queue.put(ProgressDataPacket(
                name="Done",
                device=gpu_id,
                new_task=True
            ))
        except queue.Empty:
            logging_queue.put(ProgressDataPacket(
                name="No more tasks",
                device=gpu_id,
                new_task=False
            ))
            break


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
        mp.set_start_method('spawn')
        self.experiment_queue = mp.Queue()
        self.logging_queue = mp.Queue()
        self.total_num_tasks = 0
        for experiment_name, experiment_encoder in experiments.items():
            for task_name, task in tasks.items():
                experiment = copy.deepcopy(task)
                experiment["task_name"] = task_name
                experiment["experiment_name"] = experiment_name
                experiment["encoder"] = experiment_encoder
                experiment["train_encoder"] = train_encoder
                experiment["num_workers"] = total_num_workers // self.num_threads
                self.experiment_queue.put(experiment)
                self.total_num_tasks += 1

    def run(self):
        try:
            stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.curs_set(0)

            lidx = 0
            stdscr.addstr(lidx, 0, "+" + "-"*99 + "+")

            lidx += 1
            stdscr.addstr(lidx, 0, "|" + " "*99 + "|")

            lidx += 1
            stdscr.addstr(lidx, 0, "|" + "-"*99 + "|")

            lidx += 1
            stdscr.addstr(lidx, 0, "|")
            stdscr.addstr(lidx, 2, "Device")
            stdscr.addstr(lidx, 15, "|")
            stdscr.addstr(lidx, 17, "Task")
            stdscr.addstr(lidx, 75, "|")
            stdscr.addstr(lidx, 77, "Progress")
            stdscr.addstr(lidx, 90, "|")
            stdscr.addstr(lidx, 92, "ETA")
            stdscr.addstr(lidx, 100, "|")

            lidx += 1
            stdscr.addstr(lidx, 0, "|")
            stdscr.addstr(lidx, 1, "-" * 99)
            stdscr.addstr(lidx, 100, "|")

            for _ in range(len(GPU_IDS)):
                lidx += 1
                stdscr.addstr(lidx, 0, "|" + " "*14 + "|" + " "*59 + "|" + " "*14 + "|" + " "*9 + "|")
                lidx += 1
                stdscr.addstr(lidx, 0, "|" + "-"*99 + "|")

            stdscr.addstr(lidx, 0, "+" + "-"*99 + "+")
            stdscr.refresh()

            for device_id in GPU_IDS:
                p = mp.Process(
                    target=thread_loop,
                    args=(device_id, self.experiment_queue, self.logging_queue),
                    daemon=False
                )
                p.start()

            pending_tasks = self.total_num_tasks
            while pending_tasks > 0:
                stdscr.refresh()
                stdscr.addstr(1, 5, "pyVTAB")
                stdscr.addstr(1, 30, "%s" % datetime.now().strftime("%H:%M:%S"))
                stdscr.addstr(1, 60, " Number of Tasks Completed %d/%d" % (
                    self.total_num_tasks - pending_tasks,
                    self.total_num_tasks
                ))
                try:
                    data = self.logging_queue.get(timeout=1)
                except queue.Empty:
                    data = None
                if data is not None:
                    stdscr.refresh()
                    if data.new_task:
                        pending_tasks -= 1
                    lidx = 5 if data.device == "cpu" else 5 + 2 * int(data.device[-1])
                    stdscr.addstr(lidx, 0, "|" + " " * 14 + "|" + " " * 59 + "|" + " " * 14 + "|" + " " * 9 + "|")
                    stdscr.addstr(lidx, 2, data.device)
                    stdscr.addstr(lidx, 17, data.name)
                    if data.idx is not None:
                        stdscr.addstr(lidx, 77, str(data.idx)+'/'+str(data.total))
                    if data.time_per_iter is not None:
                        mins = int((data.time_per_iter * (data.total - data.idx)) // 60)
                        secs = int((data.time_per_iter * (data.total - data.idx)) % 60)
                        stdscr.addstr(lidx, 92, "%02d:%02d" % (mins, secs))
                    stdscr.refresh()
        except:
            traceback.print_exc()
        finally:
            stdscr.keypad(0)
            curses.echo()
            curses.nocbreak()
            curses.endwin()
