import math
import torch
from .geometry import build_rotation

def __split_by_uniform(xyz, scaling, rotation, N=2, scaling_factor=0.5):
    """
        split the gaussians along the longest axis to N

        xyz: (P, 3)
        scaling: (P, 3)
        rotation: (P, 4)
        return:
            center_new: (P, N, 3)
            scaling_new: (P, N, 3)
    """
    rots = build_rotation(rotation)[:, None].repeat(1, N, 1, 1)
    samples = torch.zeros_like(xyz)
    indices = scaling.max(dim=-1).indices
    rangei = torch.arange(samples.shape[0], device=samples.device)
    samples[rangei, indices] = 1.
    steps_center = torch.tensor([-0.5, 0.5], device=samples.device)
    offset = steps_center.reshape(1, N, 1)
    samples = samples[:, None] * offset
    offset = samples * scaling[:, None]
    # calculate the rotation center
    center_old = xyz[:, None]
    center_new = torch.matmul(rots, offset.unsqueeze(-1)).squeeze(-1) + center_old
    scaling_new = scaling.detach().clone()
    scaling_new[rangei, indices] *= scaling_factor
    scaling_new = scaling_new[:, None].repeat(1, N, 1)
    return center_new, scaling_new

def _split_by_uniform(xyz, scaling, rotation, N=2, scaling_factor=0.5):
    """
        split the gaussians along the longest axis to N

        xyz: (P, nFrames, 3)
        scaling: (P, 3)
        rotation: (P, nFrames, 4)
        return:
            center_new: (P, N, 3)
            scaling_new: (P, N, 3)
    """
    if len(xyz.shape) == 3:
        nFrames = xyz.shape[1]
        xyz_ = xyz.reshape(-1, 3)
        rotation_ = rotation.reshape(-1, 4)
        scaling_ = scaling[:, None].repeat(1, nFrames, 1).reshape(-1, 3)
        center_new, scaling_new = __split_by_uniform(xyz_, scaling_, rotation_, N, scaling_factor)
        center_new = center_new.reshape(-1, nFrames, N, 3)
        scaling_new = scaling_new.reshape(-1, nFrames, N, 3)
        center_new = center_new.transpose(1, 2)
        scaling_new = scaling_new.transpose(1, 2)
        return center_new, scaling_new
    else:
        return __split_by_uniform(xyz, scaling, rotation, N, scaling_factor)

# def split_by_sample(self, flag, N=2, scaling_factor=1, reshape=True):
def split_by_sample(model, flag, N=2, reshape=True, scaling_factor=1, scaling_method='decay', depth_add=1):
    if flag.sum() == 0:
        return {}
    index = torch.where(flag)[0]
    center = model.xyz[index].detach()
    scaling = model.get_scaling_by_flag(index).detach()
    rotation = model.get_rotation_by_flag(index).detach()
    # Extract points that satisfy the gradient condition
    stds = scaling[:, None].repeat(1, N, 3//scaling.shape[-1])
    means = torch.zeros_like(stds)
    samples = torch.normal(mean=means, std=stds/scaling_factor)
    rots = build_rotation(rotation)[:, None].repeat(1, N, 1, 1)
    center_old = center[:, None]
    center_new = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + center_old
    if scaling_method == 'keep':
        new_scaling = model.scaling[flag][:, None].repeat(1, N, 1)
    elif scaling_method == 'decay':
        scaling_copy = scaling.clone()
        indices = scaling_copy.max(dim=-1).indices
        scaling_copy /= math.sqrt(N)
        scaling_new = scaling_copy[:, None].repeat(1, N, 1)
        index_repeat = index[:, None].repeat(1, N)
        new_scaling = model.inverse_scaling(scaling_new, index_repeat, depth_add=depth_add)
    elif scaling_method == 'zero':
        new_scaling = torch.zeros_like(scaling[:, None].repeat(1, N, 1))
    else:
        raise ValueError(f'Unknown scaling method {scaling_method}')
    if reshape:
        center_new = center_new.reshape(-1, 3)
        new_scaling = new_scaling.reshape(-1, new_scaling.shape[-1])
    return {
        'xyz': center_new,
        'scaling': new_scaling,
        'N': N
    }

def split_by_uniform(model, flag, N=2, reshape=True, scaling_factor=0.5, depth_add=1):
    if flag.sum() == 0:
        return {}
    index = torch.where(flag)[0]
    center = model.xyz[index].detach()
    scaling = model.get_scaling_by_flag(index)
    scaling_mean = scaling.mean().item()
    scaling_max_mean = scaling.max(dim=-1).values.mean().item()
    rotation = model.rotation[index].detach()
    # Extract points that satisfy the gradient condition
    for log2 in range(1, 4):
        center_new, new_scaling = _split_by_uniform(center, scaling, rotation, 2, scaling_factor)
        # TODO: split more time
        if reshape:
            center_new = center_new.reshape(-1, *center.shape[1:])
            new_scaling = new_scaling.reshape(-1, *scaling.shape[1:])
        if len(rotation.shape) == 2:
            rotation = rotation[:, None].repeat(1, 2, 1).reshape(-1, *rotation.shape[1:])
        else:
            rotation = rotation[:, None].repeat(1, 2, 1, 1).reshape(-1, *rotation.shape[1:])
        center, scaling = center_new, new_scaling
        if 2 ** log2 >= N:
            break
    # check the split
    print(f'[{model.__class__.__name__}] split : {flag.sum()} -> {center.shape[0]}')
    print(f'[{model.__class__.__name__}] radius: {scaling_mean:.4f} -> {new_scaling.mean().item():.4f}, {scaling_max_mean:.4f} -> {new_scaling.max(dim=-1).values.mean().item():.4f}')
    # new_scaling = model.activation.scaling_inverse_activation(new_scaling)       
    # new_scaling = model.activation.scaling_inverse_activation(new_scaling)
    index_repeat = index[:, None].repeat(1, N).reshape(-1)
    new_scaling = model.inverse_scaling(new_scaling, index_repeat, depth_add=depth_add)
    # new_scaling = model.inverse_scaling(new_scaling)
    return {
        'xyz': center_new,
        'scaling': new_scaling,
        'N': N
    }

class Splitter():
    def __init__(self, N=4, scaling_factor=0.7, split_method='uniform') -> None:
        self.N = N
        self.split_method = split_method
        self.scaling_factor = scaling_factor
    
    def split_and_remove(self, model, optimizer, flag_split, flag_remove, remove_split=True, **kwargs):
        print(f'[{self.__class__.__name__}] split method {self.split_method}, remove {flag_split.shape[0]} +{flag_split.sum()}x{self.N} -{flag_remove.sum()}')
        if self.split_method == 'uniform':
            split_info = split_by_uniform(model, flag_split, N=self.N, **kwargs)
        elif self.split_method == 'sample':
            split_info = split_by_sample(model, flag_split, N=self.N, **kwargs)
        else:
            raise ValueError(f'Unknown split method {self.split_method}')
        if remove_split:
            flag_remove = flag_remove | flag_split
        copy_device = torch.device('cpu')
        num_keep = (~flag_remove).sum()
        flag_remove = flag_remove.to(copy_device)
        flag_split = flag_split.to(copy_device)
        for key in model.keys:
            old_val = getattr(model, key, None)
            if old_val is None or old_val.shape[0] == 0:
                continue
            is_param = old_val.requires_grad
            if is_param:
                old_val = old_val.data
            origin_device = old_val.device
            old_val = old_val.to(copy_device)
            new_val = []
            # keep origin data
            new_val.append(old_val[~flag_remove])
            # append the split
            if flag_split.sum() > 0:
                if key in split_info:
                    new_val.append(split_info[key].to(copy_device))
                else:
                    N = split_info['N']
                    _old = old_val[flag_split][:, None]
                    _old = _old.repeat(1, N, *[1 for _ in range(len(old_val.shape)-1)])
                    _old = _old.reshape(-1, *old_val.shape[1:])
                    new_val.append(_old)
            new_val = torch.cat(new_val, dim=0).to(origin_device)
            if key == 'tree_depth':
                new_val[:num_keep] = old_val[~flag_remove]
                new_val[num_keep:] = old_val[flag_split][:, None].repeat(1, self.N).reshape(-1) + 1
            getattr(model, key).set_(new_val)
            # update state
        # update the state dict
        if optimizer is None:
            return num_keep
        index_keep = torch.where(~flag_remove)[0]
        for state_key in optimizer.state_keys:
            index = index_keep
            state = getattr(optimizer, state_key)
            origin_device = state.xyz.device
            state.to(torch.device('cpu'))
            if index.device != state.device:
                index = index.cpu()
            for key, val in state.items():
                new_val = getattr(model, key)
                new_zeros = torch.zeros((new_val.shape[0]-num_keep, *new_val.shape[1:]), device=val.device, dtype=val.dtype)
                _new = torch.cat([val[index], new_zeros], dim=0)
                state[key].set_(_new)
            # recover
            state.to(origin_device)
        thres_exp_avg_sq = 50_000_000 # 50M
        if model.xyz.shape[0] > thres_exp_avg_sq and optimizer.exp_avg.device != torch.device('cpu'):
            print(f'[{self.__class__.__name__}] num points {model.xyz.shape[0]} > {thres_exp_avg_sq}, move exp_avg_sq to CPU')
            optimizer.exp_avg_sq.to(torch.device('cpu'))
        if model.xyz.shape[0] > thres_exp_avg_sq * 2:
            print(f'[{self.__class__.__name__}] num points {model.xyz.shape[0]} > {thres_exp_avg_sq * 2}, move exp_avg to CPU')
            optimizer.exp_avg.to(torch.device('cpu'))
        return num_keep
    
    def split_and_remove_other(self, model, keys, flag_split, flag_remove, remove_split=True):
        if remove_split:
            flag_remove = flag_remove | flag_split
        num_keep = (~flag_remove).sum()
        for key in keys:
            old_val = getattr(model, key, None)
            if old_val is None or old_val.shape[0] == 0:
                continue
            new_val = torch.zeros((num_keep+flag_split.sum()*self.N,), device=old_val.device, dtype=old_val.dtype)
            new_val[:num_keep] = old_val[~flag_remove]
            if key == 'radius3d_min':
                new_val_copy = old_val[flag_split][:, None].repeat(1, self.N).reshape(-1)
                new_val[num_keep:] = new_val_copy
            getattr(model, key).set_(new_val)