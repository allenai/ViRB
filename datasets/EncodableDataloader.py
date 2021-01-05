import torch


from utils.progress_iterator import ProgressIterator


class EncodableDataloader:

    def __init__(
            self,
            dataloader,
            model,
            progress_name,
            logging_queue,
            batch_size=32,
            shuffle=True,
            device="cpu",
            num_dataset_repeats=1,
    ):
        model = model.to(device)
        model.eval()
        data_stacks = {name: [] for name in model.required_encoding()}
        label_stack = []
        with torch.no_grad():
            for k in range(num_dataset_repeats):
                for d, l in ProgressIterator(dataloader, progress_name+" - Pass %d" % (k+1), logging_queue, device):
                    d = d.to(device)
                    o = model.encoder_forward(d)
                    for name, data_stack in data_stacks.items():
                        data_stack.append(o[name].detach().half().cpu())
                    label_stack.append(l)
            self.data = {name: torch.cat(data_stacks[name], dim=0).half().to(device) for name in data_stacks}
            self.labels = torch.cat(label_stack, dim=0).to(device)
            del data_stacks
            del label_stack
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.batch_index = 0
        self.batch_idxs = None

    def __iter__(self):
        if self.shuffle:
            idxs = torch.randperm(self.__len__()).to(self.device)
        else:
            idxs = torch.arange(self.__len__()).to(self.device)
        self.batch_idxs = [
            idxs[i*self.batch_size:min((i+1)*self.batch_size, self.__len__())]
            for i in range(self.__len__() // self.batch_size)
        ]
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index >= len(self.batch_idxs):
            raise StopIteration
        bi = self.batch_idxs[self.batch_index]
        data = {name: self.data[name][bi] for name in self.data}, self.labels[bi]
        self.batch_index += 1
        return data

    def __len__(self):
        return self.data[list(self.data.keys())[0]].size(0)
