import tqdm
import numpy as np
import torch


class VTABTask:

    def __init__(
            self,
            name,
            encoder,
            head,
            train_set,
            test_set,
            loss,
            error,
            optimizer,
            out_dir,
            batch_size=256,
            num_workers=12
    ):
        self.task_name = name
        self.loss = loss
        self.error = error
        self.batch_size = batch_size
        self.out_dir = out_dir
        print("---------------- %s ----------------" % self.task_name)
        print("Encoding data")
        train_set.encode(encoder)
        self.train_dataloader = torch.utils.data.DataLoader(train_set,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            num_workers=num_workers)
        test_set.encode(encoder)
        self.test_dataloader = torch.utils.data.DataLoader(test_set,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           num_workers=num_workers)
        self.model = head
        self.optimizer = optimizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, epochs):
        print("Training %s" % self.task_name)
        for _ in tqdm.tqdm(range(epochs)):
            train_loss, train_error = self.train_epoch()
            print(train_loss)
        test_loss, test_accuracy = self.test()
        print("Test Result: %.4f" % test_accuracy)
        torch.save(self.model, self.out_dir)
        print("Saved model to %s\n\n\n" % self.out_dir)

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
            train_errors.append(train_error)
        return np.mean(train_losses), np.mean(train_errors)

    def test(self):
        self.model.eval()
        test_losses = []
        test_errors = []
        for x, label in self.test_dataloader:
            x, label = x.to(self.device), label.to(self.device)
            out = self.model(x)
            test_loss = self.loss(out, label)
            test_losses.append(test_loss)
            test_error = self.error(out, label)
            test_errors.append(test_error)
        return np.mean(test_losses), np.mean(test_errors)
