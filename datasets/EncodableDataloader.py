import torch
import tqdm


class EncodableDataloader:

    def __init__(self, dataloader, model, batch_size=32, shuffle=True, device="cpu"):
        model = model.to(device)
        model.eval()
        data_stacks = {name: [] for name in model.required_encoding()}
        label_stack = []
        print("Encoding Data")
        for d, l in tqdm.tqdm(dataloader):
            d = d.to(device)
            with torch.no_grad():
                o = model.encoder_forward(d)
                for name, data_stack in data_stacks.items():
                    data_stack.append(o[name].detach())
                # total_size = 0
                # for name, data_stack in data_stacks.items():
                #     for tensor in data_stack:
                #         total_size += tensor.element_size() * tensor.nelement()
                # import time
                # time.sleep(10)
                # print("Size of stack: %.4f GB" % (total_size/1.0e9))
            label_stack.append(l)
        self.data = {name: torch.cat(data_stacks[name], dim=0).to(device) for name in data_stacks}
        self.labels = torch.cat(label_stack, dim=0).to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def __iter__(self):
        if self.shuffle:
            idxs = torch.randperm(self.__len__()).to(self.device)
        else:
            idxs = torch.arange(self.__len__()).to(self.device)
        batch_idxs = [idxs[i:min(i+self.batch_size, self.__len__())] for i in range(self.__len__() // self.batch_size)]
        return iter([({name: self.data[name][bi] for name in self.data}, self.labels[bi]) for bi in batch_idxs])

    def __len__(self):
        return self.data[list(self.data.keys())[0]].size(0)
