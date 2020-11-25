import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import json


class VTABTask:

    def __init__(
            self,
            name,
            task,
            model,
            train_set,
            test_set,
            loss,
            error,
            optimizer,
            out_dir,
            scheduler,
            batch_size=512,
            num_workers=12,
            device="cpu"
    ):
        self.name = name
        self.task = task
        self.loss = loss
        self.error = error
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.train_dataloader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=num_workers)
        self.test_dataloader = torch.utils.data.DataLoader(test_set,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           num_workers=num_workers)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model.to(self.device)

    def run(self, epochs):
        os.makedirs(self.out_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=self.out_dir)
        for e in range(epochs):
            train_loss, train_accuracy = self.train_epoch()
            test_loss, test_accuracy = self.test()
            writer.add_scalar("TrainLoss/"+self.task, train_loss, e)
            writer.add_scalar("TestAccuracy/"+self.task, test_accuracy, e)
            if self.scheduler:
                self.scheduler.step()
        test_loss, test_accuracy = self.test()
        data = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        }
        return data
        # with open(self.out_dir+"/results.json", "w") as f:
        #     json.dump(data, f)
        # torch.save(self.model, self.out_dir+"/model.pt")

    def train_epoch(self):
        self.model.train()
        train_losses = []
        train_errors = []
        for x, label in self.train_dataloader:
            x, label = x.to(self.device), label.to(self.device)
            self.model.zero_grad()
            out = self.model(x)
            train_loss = self.loss(out, label)
            train_loss.backward()
            train_losses.append(train_loss.item())
            self.optimizer.step()
            train_error = self.error(out, label)
            train_errors.append(train_error.item())
        return np.mean(train_losses), np.mean(train_errors)

    def test(self):
        self.model.eval()
        test_losses = []
        test_errors = []
        for x, label in self.test_dataloader:
            x, label = x.to(self.device), label.to(self.device)
            with torch.no_grad():
                out = self.model(x)
                test_loss = self.loss(out, label)
                test_losses.append(test_loss.item())
                test_error = self.error(out, label)
                test_errors.append(test_error.item())
        return np.mean(test_losses), np.mean(test_errors)
