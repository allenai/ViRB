import time
import numpy as np

from utils.progress_data_packets import ProgressDataPacket


class ProgressIterator:

    def __init__(self, iterator, name, queue, device):
        self.data = iterator
        self.iter = iterator.__iter__()
        self.name = name
        self.queue = queue
        self.device = device
        self.idx = 0
        self.start_time = None
        self.eta_list = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.data):
            raise StopIteration
        if self.start_time is None:
            delta = None
            self.start_time = time.time()
        else:
            end_time = time.time()
            self.eta_list.append(end_time - self.start_time)
            self.start_time = end_time
            delta = np.mean(self.eta_list)
        self.queue.put(ProgressDataPacket(
            name=self.name,
            device=self.device,
            idx=self.idx+1,
            total=len(self.data),
            delta_time=delta)
        )
        item = self.iter.__next__()
        self.idx += 1
        return item
