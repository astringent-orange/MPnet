class Trainer(object):
    def __init__(self, opt, model, optimizer):
        self.opt = opt
        self.model = model
        self.optimizer = optimizer

    def set_device(self, gpu_id, device):
        self.gpu_id = gpu_id
        self.device = device
        self.model.to(self.device)
        