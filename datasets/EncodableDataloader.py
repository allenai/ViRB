import torch


from utils.progress_iterator import ProgressIterator


class EncodableDataloader:

    def __init__(
            self,
            dataloader,
            model,
            name,
            logging_queue,
            batch_size=32,
            shuffle=True,
            device="cpu",
            principal_directions=None
    ):
        self.principal_directions = principal_directions
        model = model.to(device)
        model.eval()
        data_stacks = {name: [] for name in model.required_encoding()}
        label_stack = []
        with torch.no_grad():
            for d, l in ProgressIterator(dataloader, name, logging_queue, device):
                d = d.to(device)
                o = model.encoder_forward(d)
                for name, data_stack in data_stacks.items():
                    if model.pca_embeddings() is not None and name in model.pca_embeddings():
                        with torch.no_grad():
                            x = o[name].detach()
                            if self.principal_directions is None:
                                self.principal_directions = {}
                            if name not in self.principal_directions:
                                self.principal_directions[name] = get_principal_directions(
                                    x, model.pca_embeddings()[name]
                                )
                            data_stack.append(get_principal_components(x, self.principal_directions[name]).half().cpu())
                    else:
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
        # import gc
        # import json
        # print("\n"* 50)
        # tensors = []
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             tensors.append({
        #                 "size": str(obj.shape),
        #                 "device": str(obj.device)
        #             })
        #     except:
        #         pass
        # print(tensors)
        # with open("tensor_dump.json", "w") as f:
        #     json.dump(tensors, f)
        # exit()

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
        #iter([({name: self.data[name][bi] for name in self.data}, self.labels[bi]) for bi in batch_idxs])

    def __next__(self):
        if self.batch_index >= len(self.batch_idxs):
            raise StopIteration
        bi = self.batch_idxs[self.batch_index]
        data = {name: self.data[name][bi] for name in self.data}, self.labels[bi]
        self.batch_index += 1
        return data

    def __len__(self):
        return self.data[list(self.data.keys())[0]].size(0)

    def get_principal_directions(self):
        return self.principal_directions


def get_principal_directions(x, num_dims):
    q = min(x.size(1), 1024)
    if len(x.shape) == 4:
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, x.size(3))
    _, _, V = torch.pca_lowrank(x, q=q, niter=10)
    return V[:, :num_dims]


def get_principal_components(x, V):
    if len(x.shape) == 4:
        x = x.permute(0, 2, 3, 1)
        original_shape = x.shape
        x = x.reshape(-1, x.size(3))
    res = torch.matmul(x, V)
    if len(original_shape) == 4:
        res = res.reshape(original_shape[0], original_shape[1], original_shape[2], -1)
        res = res.permute(0, 3, 1, 2)
    return res
