import numpy as np
import torch
import torch.nn as nn
import math

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def _single_tensor_adam(param,
                        grad,
                        exp_avg,
                        exp_avg_sq,
                        max_exp_avg_sq,
                        step_t,
                        lr,
                        eps: float,
                        beta1=0.9,
                        beta2=0.999,
                        ):
    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
    if torch.is_tensor(step_t):
        if len(grad.shape) == 2:
            step = step_t[:, None]
        elif len(grad.shape) == 3:
            step = step_t[:, None, None]
        else:
            step = step_t
    else:
        step = int(step_t)
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = lr / bias_correction1
    if torch.is_tensor(step):
        bias_correction2_sqrt = torch.sqrt(bias_correction2)
    else:
        bias_correction2_sqrt = math.sqrt(bias_correction2)
    if max_exp_avg_sq is not None:
        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
    else:
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
    param.data.add_(-step_size * (exp_avg / denom))
    # param.addcdiv_(exp_avg, denom, value=-step_size)
    return param, exp_avg, exp_avg_sq, max_exp_avg_sq

class State(nn.Module):
    def __init__(self, exp_avg, exp_avg_sq, max_exp_avg_sq, steps) -> None:
        super().__init__()
        self.register_buffer('exp_avg', exp_avg)
        self.register_buffer('exp_avg_sq', exp_avg_sq)
        if max_exp_avg_sq is not None:
            self.register_buffer('max_exp_avg_sq', max_exp_avg_sq)
        self.register_buffer('steps', steps)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __repr__(self):
        return f'State: {self.exp_avg.shape} steps: {self.steps.float().mean()}'

class BufferDict(nn.Module):
    def __init__(self, dicts):
        super().__init__()
        for key, value in dicts.items():
            self.register_buffer(key, value)
        self.keys = list(dicts.keys())
    
    @property
    def device(self):
        return self[self.keys[0]].device

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __repr__(self):
        return f'BufferDict: {self.keys}'
    
    def items(self):
        for key in self.keys:
            yield key, getattr(self, key)
    
    def get(self, key, default_value=None):
        if key not in self.keys:
            return default_value
        return getattr(self, key)

class SparseOptimizer(nn.Module):
    def __init__(self, optimize_keys, lr_dict, model, device=None, xyz_scale=None, use_amsgrad=False) -> None:
        super().__init__()
        if device is None:
            device = model.xyz.device
        self.register_buffer('global_steps', torch.tensor(0, dtype=torch.float32, device=device))
        self.device = device
        self.optimize_keys = optimize_keys
        self.state_keys = ['exp_avg', 'exp_avg_sq']
        self.lr_dict = lr_dict
        # add steps
        exp_avg, exp_avg_sq, max_exp_avg_sq = {}, {}, {}
        steps = {}
        for key in optimize_keys:
            param = getattr(model, key)
            exp_avg_key = torch.zeros_like(param, device=device)
            exp_avg[key] = exp_avg_key
            exp_avg_sq[key] = exp_avg_key.clone()
            if use_amsgrad:
                max_exp_avg_sq[key] = exp_avg_key.clone()
            steps[key] = torch.zeros((param.shape[0], ), dtype=torch.int32, device=device)
        self.exp_avg = BufferDict(exp_avg)
        self.exp_avg_sq = BufferDict(exp_avg_sq)
        if use_amsgrad:
            self.max_exp_avg_sq = BufferDict(max_exp_avg_sq)
            self.state_keys.append('max_exp_avg_sq')
        self.use_amsgrad = use_amsgrad
        self.steps = BufferDict(steps)
        if xyz_scale is not None:
            print(f'[{self.__class__.__name__}] xyz_scale: {xyz_scale}, steps: {lr_dict["max_steps"]}, lr {lr_dict["xyz"] * xyz_scale}->{lr_dict["xyz_final"] * xyz_scale}')
            self.xyz_scheduler_args = get_expon_lr_func(
                lr_init=lr_dict['xyz'] * xyz_scale,
                lr_final=lr_dict.get('xyz_final', lr_dict['xyz'] * 0.01) * xyz_scale,
                max_steps=int(lr_dict['max_steps']))
            if 'scaling' in lr_dict:
                print(f'[{self.__class__.__name__}] scaling: {lr_dict["scaling"]} -> {lr_dict.get("scaling_final", lr_dict["scaling"] * 1)}')
                self.scaling_scheduler_args = get_expon_lr_func(
                    lr_init=lr_dict['scaling'],
                    lr_final=lr_dict.get('scaling_final', lr_dict['scaling'] * 1) ,
                    max_steps=int(lr_dict['max_steps']))
            self.xyz_lr = lr_dict['xyz'] * xyz_scale

    def step(self, model, index, params, flag_vis):
        self.global_steps += 1
        # only step the selecte tensors
        # this is sparse tensor
        index = index[flag_vis]
        index_cpu = index.cpu()
        # select the optimizer state and steps
        exp_avg, exp_avg_sq, max_exp_avg_sq = self.index_select_optimizer(index, index_cpu, params['xyz'].device)
        for key, param in params.items():
            if param.grad is None:
                continue
            if key == 'xyz':
                lr = self.xyz_scheduler_args(self.global_steps.item())
                self.xyz_lr = lr
            elif key == 'scaling':
                lr = self.scaling_scheduler_args(self.global_steps.item())
            else:
                lr = self.lr_dict[key]
            param, exp_avg_, exp_avg_sq_, max_exp_avg_sq_ = _single_tensor_adam(
                param.data[flag_vis],
                param.grad[flag_vis],
                exp_avg[key],
                exp_avg_sq[key],
                max_exp_avg_sq.get(key, None),
                # self.optimizer.state[key].steps[index],
                self.global_steps.item(),
                lr,
                eps=1e-15
            )
            # update
            getattr(model, key).data[index] = param.data
            # update to origin state
        self.index_update_optimizer(index, index_cpu,
            exp_avg, exp_avg_sq, max_exp_avg_sq)

    def index_select_optimizer(self, index, index_cpu, target_device):
        exp_avg_index, exp_avg_sq_index, max_exp_avg_sq_index = {}, {}, {}
        for key in self.exp_avg.keys:
            if self.exp_avg[key].device != target_device: # val in the CPU
                exp_avg_index[key] = self.exp_avg[key][index_cpu]
            else:
                exp_avg_index[key] = self.exp_avg[key][index]
            if self.exp_avg_sq[key].device != target_device:
                exp_avg_sq_index[key] = self.exp_avg_sq[key][index_cpu]
            else:
                exp_avg_sq_index[key] = self.exp_avg_sq[key][index]
            if self.use_amsgrad:
                if self.max_exp_avg_sq[key].device != target_device:
                    max_exp_avg_sq_index[key] = self.max_exp_avg_sq[key][index_cpu]
                else:
                    max_exp_avg_sq_index[key] = self.max_exp_avg_sq[key][index]
        exp_avg_index = BufferDict(exp_avg_index)
        exp_avg_sq_index = BufferDict(exp_avg_sq_index)
        exp_avg_index.to(target_device, non_blocking=True)
        exp_avg_sq_index.to(target_device, non_blocking=True)
        if self.use_amsgrad:
            max_exp_avg_sq_index = BufferDict(max_exp_avg_sq_index)
            max_exp_avg_sq_index.to(target_device, non_blocking=True)
        # to device
        return exp_avg_index, exp_avg_sq_index, max_exp_avg_sq_index
    
    def index_update_optimizer(self, index, index_cpu,
                exp_avg, exp_avg_sq, max_exp_avg_sq):
        # update the parameter to origin device
        if exp_avg.device != self.exp_avg.device:
            exp_avg = exp_avg.to(self.exp_avg.device, non_blocking=True)
        if exp_avg_sq.device != self.exp_avg_sq.device:
            exp_avg_sq = exp_avg_sq.to(self.exp_avg_sq.device, non_blocking=True)
        if self.use_amsgrad:
            if max_exp_avg_sq.device != self.max_exp_avg_sq.device:
                max_exp_avg_sq = max_exp_avg_sq.to(self.max_exp_avg_sq.device, non_blocking=True)
        # update
        for key in self.exp_avg.keys:
            if self.exp_avg[key].device == index_cpu.device:
                self.exp_avg[key][index_cpu] = exp_avg[key]
            else:
                self.exp_avg[key][index] = exp_avg[key]
            if self.exp_avg_sq[key].device != index_cpu.device:
                self.exp_avg_sq[key][index_cpu] = exp_avg_sq[key]
            else:
                self.exp_avg_sq[key][index] = exp_avg_sq[key]
            if self.use_amsgrad:
                if self.max_exp_avg_sq[key].device != index_cpu.device:
                    self.max_exp_avg_sq[key][index_cpu] = max_exp_avg_sq[key]
                else:
                    self.max_exp_avg_sq[key][index] = max_exp_avg_sq[key]

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        print(f'[{self.__class__.__name__}] load state dict')
        print(f'[{self.__class__.__name__}] global steps: {self.global_steps}')