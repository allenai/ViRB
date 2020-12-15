import time

from utils.progress_data_packets import ProgressDataPacket


class ProgressIterator:

    def __init__(self, iterator, name, queue, device):
        self.iter = list(iterator)
        self.name = name
        self.queue = queue
        self.device = device
        self.idx = 0
        self.start_time = time.time()

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.iter):
            raise StopIteration
        end_time = time.time()
        delta = end_time - self.start_time
        self.start_time = end_time
        self.queue.put(ProgressDataPacket(
            name=self.name,
            device=self.device,
            idx=self.idx+1,
            total=len(self.iter),
            delta_time=delta)
        )
        item = self.iter[self.idx]
        self.idx += 1
        return item
