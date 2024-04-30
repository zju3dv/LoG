import numpy as np
import torch
import torch.nn as nn
from .sparse_optimizer import SparseOptimizer, _single_tensor_adam

class Corrector(nn.Module):
    def __init__(self, use_view_correction, start_step=1000, lr_init=0.1, lr_final=0.001):
        super().__init__()
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.start_step = start_step
        self.use_view_correction = use_view_correction
        view_correction = torch.ones((0, 3))
        self.view_correction = nn.Parameter(view_correction)
        self.register_buffer('xyz', view_correction)
        self.use_cpu_adam = False
        self.use_amsgrad = True
        self.index = None
    
    def init(self, num_views):
        if self.use_view_correction:
            view_correction = torch.ones((num_views, 3), device=self.view_correction.device)
            self.view_correction.data = view_correction
            print(f'[{self.__class__.__name__}] init view correction: {num_views}')
    
    def training_setup(self):
        if getattr(self, 'optimizer', None) is not None:
            print(f'[{self.__class__.__name__}] optimizer is already setup')
            return 0
        if self.use_view_correction:
            lr = 1e-3
            self.optimizer = SparseOptimizer(['view_correction'], {'view_correction': lr}, self, use_amsgrad=self.use_amsgrad)
            print(f'[{self.__class__.__name__}] view correction optimizer setup {lr}')

    def step(self):
        if not self.use_view_correction:
            return 0
        index = [self.index]
        self.optimizer.steps['view_correction'][index] += 1
        steps = self.optimizer.steps['view_correction'][index] - self.start_step
        if steps < 0:
            return 0
        exp_avg = self.optimizer.exp_avg['view_correction'][index]
        exp_avg_sq = self.optimizer.exp_avg_sq['view_correction'][index]
        max_exp_avg_sq = self.optimizer.max_exp_avg_sq['view_correction'][index]
        lr_init, lr_final = self.lr_init, self.lr_final
        max_steps = 100
        t = np.clip(steps.item() / max_steps, 0, 1)
        lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        param = self.view_correction
        param, exp_avg, exp_avg_sq, max_exp_avg_sq = _single_tensor_adam(
            param.data[index], param.grad[index],
            exp_avg, exp_avg_sq, max_exp_avg_sq,
            steps, lr, eps=1e-15)
        # update the parameter
        self.view_correction.data[index] = param
        # zero grad
        self.view_correction.grad[index] = 0
        # update the state dict
        self.optimizer.exp_avg['view_correction'][index] = exp_avg
        self.optimizer.exp_avg_sq['view_correction'][index] = exp_avg_sq
        self.optimizer.max_exp_avg_sq['view_correction'][index] = max_exp_avg_sq
    
    def __getitem__(self, index):
        self.index = index
        return self.view_correction[index]
