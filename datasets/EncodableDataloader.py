import torch
import tqdm


class EncodableDataloader:

    def __init__(self, dataloader, model, batch_size=32, shuffle=True, device="cpu", principal_directions=None):
        self.principal_directions = principal_directions
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
                    if model.pca_embeddings() is not None and name in model.pca_embeddings():
                        with torch.no_grad():
                            x = o[name].detach()
                            if self.principal_directions is None:
                                self.principal_directions = {}
                            if name not in self.principal_directions:
                                self.principal_directions[name] = get_principal_directions(
                                    x, model.pca_embeddings()[name]
                                )
                            data_stack.append(get_principal_components(x, self.principal_directions[name]).half())
                    else:
                        data_stack.append(o[name].detach().half())
                # total_size = 0
                # for name, data_stack in data_stacks.items():
                #     for tensor in data_stack:
                #         total_size += tensor.element_size() * tensor.nelement()
                # import time
                # time.sleep(10)
                # print("Size of stack: %.4f GB" % (total_size/1.0e9))
            label_stack.append(l)
        # print("data stacks!!!")
        # for name in data_stacks:
        #     print(name, len(data_stacks[name]))
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
        batch_idxs = [idxs[i*self.batch_size:min((i+1)*self.batch_size, self.__len__())] for i in range(self.__len__() // self.batch_size)]
        return iter([({name: self.data[name][bi] for name in self.data}, self.labels[bi]) for bi in batch_idxs])

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
