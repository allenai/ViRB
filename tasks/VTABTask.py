import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm
import json

from datasets.EncodableDataloader import EncodableDataloader


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

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model.to(self.device)

        self.train_dataloader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=num_workers)
        self.test_dataloader = torch.utils.data.DataLoader(test_set,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           num_workers=1)
        if not self.model.train_encoder:
            self.train_dataloader = EncodableDataloader(self.train_dataloader,
                                                        self.model,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        device=device)
            self.test_dataloader = EncodableDataloader(self.test_dataloader,
                                                        self.model,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        device=device)

    def run(self, epochs):
        print("Training %s on %s" % (self.name, self.task))
        os.makedirs(self.out_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=self.out_dir)
        print("Training")
        for e in tqdm.tqdm(range(epochs)):
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
        num_samples = 0
        if self.model.train_encoder:
            import time
            print("Starting Sleep")
            time.sleep(10)
            print("Done Sleep")
            for x, label in self.train_dataloader:
                num_samples_in_batch = x.size(0)
                num_samples += num_samples_in_batch
                x, label = x.to(self.device), label.to(self.device)
                self.model.zero_grad()
                out = self.model(x)
                train_loss = self.loss(out, label)
                train_loss.backward()
                train_losses.append(train_loss.item() * num_samples_in_batch)
                self.optimizer.step()
                train_error = self.error(out, label)
                train_errors.append(train_error.item() * num_samples_in_batch)
            return np.sum(train_losses) / num_samples, np.sum(train_errors) / num_samples
        for x, label in self.train_dataloader:
            num_samples_in_batch = x.size(0)
            num_samples += num_samples_in_batch
            self.model.zero_grad()
            out = self.model.head_forward(x)
            train_loss = self.loss(out, label)
            train_loss.backward()
            train_losses.append(train_loss.item() * num_samples_in_batch)
            self.optimizer.step()
            train_error = self.error(out, label)
            train_errors.append(train_error.item() * num_samples_in_batch)
        return np.sum(train_losses) / num_samples, np.sum(train_errors) / num_samples

    def test(self):
        self.model.eval()
        test_losses = []
        test_errors = []
        num_samples = 0
        if self.model.train_encoder:
            for x, label in self.test_dataloader:
                num_samples_in_batch = x.size(0)
                num_samples += num_samples_in_batch
                x, label = x.to(self.device), label.to(self.device)
                with torch.no_grad():
                    out = self.model(x)
                    test_loss = self.loss(out, label)
                    test_losses.append(test_loss.item() * num_samples_in_batch)
                    test_error = self.error(out, label)
                    test_errors.append(test_error.item() * num_samples_in_batch)
            return np.sum(test_losses) / num_samples, np.sum(test_errors) / num_samples
        for x, label in self.test_dataloader:
            num_samples_in_batch = x.size(0)
            num_samples += num_samples_in_batch
            with torch.no_grad():
                out = self.model.head_forward(x)
                test_loss = self.loss(out, label)
                test_losses.append(test_loss.item() * num_samples_in_batch)
                test_error = self.error(out, label)
                test_errors.append(test_error.item() * num_samples_in_batch)
        return np.sum(test_losses) / num_samples, np.sum(test_errors) / num_samples
