class ProgressDataPacket:

    def __init__(self, name=None, device=None, idx=None, total=None, delta_time=None, new_task=False):
        self.name = name
        self.device = device
        self.idx = idx
        self.total = total
        self.time_per_iter = delta_time
        self.new_task = new_task
