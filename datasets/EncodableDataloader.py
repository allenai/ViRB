import torch
import tqdm

class EncodableDataloader:

    def __init__(self, dataloader, model, batch_size=512, shuffle=True, device="cpu"):
        model = model.to(device)
        data_stacks = {name: []  for name in model.required_encoding()}
        label_stack = []
        for d, l in dataloader:
            d = d.to(device)
            with torch.no_grad:
                o = model.encoder_forward(d)
                for name, data_stack in data_stacks.items():
                    data_stack.append(o[name])
            label_stack.append(l)
        self.data = {name: torch.cat(data_stacks[name], dim=0).to(device) for name in data_stacks}
        self.labels = torch.cat(label_stack, dim=0).to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def __call__(self):
        if self.shuffle:
            idxs = torch.arange(self.__len__()).to(self.device)
        else:
            idxs = torch.randperm(self.__len__()).to(self.device)
        batch_idxs = [idxs[i:min(i+self.batch_size, self.__len__())] for i in range(self.__len__() // self.batch_size)]
        return [({name: self.data[bi].detach() for name in self.data}, self.labels[bi]) for bi in batch_idxs]

    def __len__(self):
        return self.data.size(0)
