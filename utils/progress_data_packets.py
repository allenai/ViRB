class ProgressDataPacket:

    def __init__(self, name, device, idx, total, delta_time):
        self.name = name
        self.device = device
        self.idx = idx
        self.total = total
        self.time_per_iter = delta_time
