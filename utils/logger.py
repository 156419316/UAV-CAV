from torch.utils.tensorboard import SummaryWriter

class RLLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, image, step):
        self.writer.add_image(tag, image, step, dataformats='HW')

    def close(self):
        self.writer.close()
