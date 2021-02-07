import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm
import json
import pickle

from datasets.EncodableDataloader import EncodableDataloader
from utils.progress_iterator import ProgressIterator


class VTABTask:

    def __init__(
            self,
            name,
            task,
            training_configs,
            train_set,
            test_set,
            loss,
            error,
            out_dir,
            logging_queue,
            batch_size=16,
            num_workers=12,
            device="cpu",
            pre_encode=None,
            num_dataset_repeats=1,
    ):
        if pre_encode is None:
            pre_encode = not training_configs[0]["model"].train_encoder
        self.pre_encode = pre_encode
        self.name = name
        self.task = task
        self.loss = loss
        self.error = error
        self.batch_size = batch_size
        self.out_dir_root = out_dir
        self.logging_queue = logging_queue

        self.training_configs = training_configs
        self.device = device
        self.training_configs[0]["model"].to(self.device)

        self.train_dataloader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=num_workers)
        self.test_dataloader = torch.utils.data.DataLoader(test_set,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)
        if self.pre_encode:
            self.train_dataloader = EncodableDataloader(self.train_dataloader,
                                                        self.training_configs[0]["model"],
                                                        "Encoding Train Set for %s on %s" % (self.name, self.task),
                                                        self.logging_queue,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        device=device,
                                                        num_dataset_repeats=num_dataset_repeats)
            self.test_dataloader = EncodableDataloader(
                self.test_dataloader,
                self.training_configs[0]["model"],
                "Encoding Test Set for %s on %s" % (self.name, self.task),
                self.logging_queue,
                batch_size=batch_size,
                shuffle=False,
                device=device)

    def run(self, epochs):
        for config in self.training_configs:
            out_dir = self.out_dir_root + "-" + config["name"]
            config["model"].to(self.device)
            os.makedirs(out_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=out_dir)
            for e in ProgressIterator(
                    range(epochs),
                    "Training %s on %s" % (self.name, self.task),
                    self.logging_queue, self.device
                    # "Training %s on %s with config %s" % (self.name, self.task, config["name"]),
                    # self.logging_queue, self.device
            ):
                train_loss, train_accuracy = self.train_epoch(config["model"], config["optimizer"])
                test_loss, test_accuracy = self.test(config["model"])
                writer.add_scalar("TrainLoss/"+self.task, train_loss, e)
                writer.add_scalar("TrainAccuracy/" + self.task, train_accuracy, e)
                writer.add_scalar("TestAccuracy/"+self.task, test_accuracy, e)
                if "scheduler" in config and config["scheduler"] is not None:
                    config["scheduler"].step()
                data = {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy
                }
                with open(out_dir+"/results.json", "w") as f:
                    json.dump(data, f)
                torch.save(config["model"].task_head.state_dict(), out_dir+"/model_head.pt")
            # if not config["model"].train_encoder and config["model"].pca_embeddings() is not None:
            #     principle_directions = self.train_dataloader.get_principal_directions()
            #     cpu_principle_directions = {name: pd.detach().cpu() for name, pd in principle_directions.items()}
            #     with open(out_dir+"/principle_directions.pkl", "wb") as f:
            #         pickle.dump(cpu_principle_directions, f)

    def run_test(self, test_configs):
        for config in test_configs:
            out_dir = self.out_dir_root + "-" + config["name"]
            config["model"].to(self.device)
            os.makedirs(out_dir, exist_ok=True)

            test_loss, test_accuracy = self.test(config["model"])
            data = {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy
            }
            with open(out_dir+"/results.json", "w") as f:
                json.dump(data, f)


    def train_epoch(self, model, optimizer):
        model.train()
        train_losses = []
        train_errors = []
        num_samples = 0
        if not self.pre_encode:
            for x, label in self.train_dataloader:
                num_samples_in_batch = x.size(0)
                num_samples += num_samples_in_batch
                x, label = x.to(self.device), label.to(self.device)
                model.zero_grad()
                out = model(x)
                train_loss = self.loss(out, label)
                train_loss.backward()
                train_losses.append(train_loss.item() * num_samples_in_batch)
                optimizer.step()
                train_error = self.error(out, label)
                train_errors.append(train_error.item() * num_samples_in_batch)
            return np.sum(train_losses) / num_samples, np.sum(train_errors) / num_samples
        for x, label in self.train_dataloader:
            num_samples_in_batch = x[list(x.keys())[0]].size(0)
            num_samples += num_samples_in_batch
            model.zero_grad()
            out = model.head_forward(x)
            train_loss = self.loss(out, label)
            train_loss.backward()
            train_losses.append(train_loss.item() * num_samples_in_batch)
            optimizer.step()
            train_error = self.error(out, label)
            train_errors.append(train_error.item() * num_samples_in_batch)
        return np.sum(train_losses) / num_samples, np.sum(train_errors) / num_samples

    def test(self, model):
        # model.eval()
        test_losses = []
        test_errors = []
        num_samples = 0
        if not self.pre_encode:
            for x, label in self.test_dataloader:
                num_samples_in_batch = x.size(0)
                num_samples += num_samples_in_batch
                x, label = x.to(self.device), label.to(self.device)
                with torch.no_grad():
                    out = model(x)
                    test_loss = self.loss(out, label)
                    test_losses.append(test_loss.item() * num_samples_in_batch)
                    test_error = self.error(out, label)
                    test_errors.append(test_error.item() * num_samples_in_batch)
            # import matplotlib.pyplot as plt
            # plt.imshow(out.detach().cpu().numpy().transpose(0, 2, 3, 1)[0])
            # plt.savefig("out.jpg")
            # torch.save(model.state_dict(), "random_encoder_step.pt")
            return np.sum(test_losses) / num_samples, np.sum(test_errors) / num_samples

        out = None
        label = None

        for x, label in self.test_dataloader:
            num_samples_in_batch = x[list(x.keys())[0]].size(0)
            num_samples += num_samples_in_batch
            with torch.no_grad():
                out = model.head_forward(x)
                test_loss = self.loss(out, label)
                test_losses.append(test_loss.item() * num_samples_in_batch)
                test_error = self.error(out, label)
                test_errors.append(test_error.item() * num_samples_in_batch)

        # import matplotlib.pyplot as plt
        # import random
        # ridx = random.randint(0, out.size(0)-1)
        # npout = torch.round(torch.sigmoid(out[ridx, 0])).detach().cpu()
        # plt.imshow(npout)
        # plt.savefig("prediction.png")
        # nplabel = label[ridx, 0].detach().cpu()
        # plt.imshow(nplabel)
        # plt.savefig("label.png")

        return np.sum(test_losses) / num_samples, np.sum(test_errors) / num_samples
