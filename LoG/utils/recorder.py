class Recorder(object):
    def __init__(self, log_dir) -> None:
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def log(self, step, key, val):
        self.writer.add_scalar(key, val, step)
