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
        end_time = time.time()
        delta = end_time - self.start_time
        self.queue.put(
            ProgressDataPacket(self.name, self.device, self.idx+1, len(self.iter), delta)
        )
        if self.idx >= len(self.iter):
            raise StopIteration
        item = self.iter[self.idx]
        self.idx += 1
        return item
