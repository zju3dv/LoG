import numpy as np
import torch
import torch.nn as nn
from .sh_utils import RGB2SH, SH2RGB, eval_sh_wobase
from .model_utils import get_module_by_str

rotation_activation = torch.nn.functional.normalize

class VisibleChecker:
    @staticmethod
    def _visible_flag_by_camera(xyz, camera, padding=0.05, squeeze=False):
        full_proj_transform = camera['full_proj_transform']
        if squeeze:
            full_proj_transform = full_proj_transform[0]
        xyz1 = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
        xyz1RTK = xyz1 @ full_proj_transform
        pw = 1.0 / (xyz1RTK[..., 3:4] + 1e-7)
        p_proj = xyz1RTK[:, :3] * pw

        depth = p_proj[:, 2]
        valid_flag = (depth > 0.) & (depth < 1.) & \
            (p_proj[:, 0] > -1 - padding) & (p_proj[:, 0] < 1. + padding) & \
            (p_proj[:, 1] > -1 - padding) & (p_proj[:, 1] < 1. + padding)
        return valid_flag, depth, p_proj
    
    def visible_flag_by_camera(self, xyz, camera, padding=0.05, squeeze=False):
        valid_flag, depth, p_proj = self._visible_flag_by_camera(xyz, camera, padding, squeeze)
        return valid_flag

class InitByPointCloud:
    def create_from_point(self, filename, scale3d, ret_scale=True, **kwargs):
        from ..utils.file import read_ply_and_log
        xyz, colors = read_ply_and_log(filename, scale3d, **kwargs)
        print(f'[Load PLY] load from ply: {filename}')
        print(f'[Load PLY] min: {xyz.min(axis=0)}, max: {xyz.max(axis=0)}')
        xyz = torch.FloatTensor(xyz)
        colors = torch.FloatTensor(colors)
        if ret_scale:
            from simple_knn._C import distCUDA2
            dist2 = torch.clamp_min(distCUDA2(xyz.cuda()), 1e-7) #3e-4^2
            # scales = torch.clamp(torch.sqrt(dist2), self.scale_min * 2, self.scale_max / 2).to(xyz.device)
            scales = torch.sqrt(dist2).cpu()
            print(f'[Load PLY] scale: {scales.min().item():.4f}, {scales.max().item():.4f}, mean = {scales.mean().item():.4f}')
        else:
            scales = None
        return xyz, colors, scales
    
    def register_by_pointcloud(self, xyz, colors, scales, init_opacity):
        self.scaling = nn.Parameter(self.scaling_inverse_activation(scales)[:, None].repeat(1, 3))
        _scales = self.scaling.data
        print(f'[{self.__class__.__name__}] scales ranges: [{_scales.min().item():.4f}~{_scales.mean().item():.4f}~{_scales.max().item():.4f}]')

        self.colors = nn.Parameter(RGB2SH(colors))
        # add sh
        if self.max_sh_degree > 0:
            features = torch.zeros((colors.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32)
            self.shs = nn.Parameter(features)
        self.xyz = nn.Parameter(xyz)
        opacity = torch.ones_like(xyz[:, :1]) * init_opacity
        self.opacity = nn.Parameter(self.opacity_inverse_activation(opacity))
        self.rotation = nn.Parameter(self.init_rotation(xyz.shape[0], xyz.device))

class BaseGaussian(nn.Module):
    def __init__(self, optimize_keys=[], lr_dict={}, 
                 use_amsgrad=False,
                 densify_and_remove={}):
        super().__init__()
        self.register_buffer('empty_xyz', torch.zeros((0, 3)))
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = lambda x: torch.log((x)/(1-x))
        self.optimize_keys = optimize_keys
        self.use_amsgrad = use_amsgrad
        self.state_keys = ['exp_avg', 'exp_avg_sq']
        if use_amsgrad:
            self.state_keys.append('max_exp_avg_sq')
        self.state_keys
        self.optimizer = None
        self.lr_dict = lr_dict
        self.print = lambda x: print(f'[{self.__class__.__name__}] {x}')
        self.current = {}
        self.warning = set()
        self.visibility_flag = {}
        self.densify_and_remove = densify_and_remove
        self.num_points = 0
        self.lr = 0.
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self.stage_name = 'full'

    def __repr__(self):
        return f'{self.__class__.__name__} with {self.xyz.shape[0]} points'

    @classmethod
    def create_from_record(cls, record):
        model = cls()
        model.colors = nn.Parameter(torch.FloatTensor(record['colors']))
        model.xyz = nn.Parameter(torch.FloatTensor(record['xyz']))
        model.scaling = nn.Parameter(model.scaling_inverse_activation(torch.FloatTensor(record['scaling'])))
        model.opacity = nn.Parameter(model.opacity_inverse_activation(torch.FloatTensor(record['opacity'])))
        model.rotation = nn.Parameter(torch.FloatTensor(record['rotation']))
        return model

    def once_warning(self, text):
        if text not in self.warning:
            self.print(text)
            self.warning.add(text)

    def clear(self):
        self.current = {}
        self.visibility_flag = {}

    def set_stage(self, stage):
        self.stage_name = stage

    def get_all(self, camera=None, rasterizer=None):
        return {
            'xyz': self.xyz,
            'colors': self.colors,
            'scaling': self.scaling_activation(self.scaling),
            'opacity': self.opacity_activation(self.opacity),
            'rotation': rotation_activation(self.rotation),
        }

    def get_xyz(self):
        return self.xyz
    
    def get_depth(self, camera):
        xyz = self.get_xyz()
        return self.depth_by_camera(xyz, camera, squeeze=False)

    def init_opacity(self, num_points, init_opacity, device):
        opacity = torch.zeros((num_points, 1), dtype=torch.float32, device=device) + init_opacity
        return self.opacity_inverse_activation(opacity)

    def get_opacity(self):
        self.once_warning('Using default opacity = 0.99')
        return 0.99 * torch.ones_like(self.xyz[:, :1])
    
    def get_colors(self, camera=None, flag=None):
        colors = SH2RGB(self.colors)
        if flag is not None:
            colors = colors[flag]
        if camera is None or self.active_sh_degree == 0:
            return colors
        if self.active_sh_degree > 0:
            shs = self.shs
            if flag is not None:
                shs = shs[flag]
            xyz = self.xyz.detach()
            if flag is not None:
                xyz = xyz[flag]
            dir_pp = xyz - camera['camera_center'][None]
            dir_pp = dir_pp / torch.norm(dir_pp, dim=-1, keepdim=True)
            colors = colors + eval_sh_wobase(dir_pp, shs, self.active_sh_degree)
        return colors

    def get_scaling(self):
        self.once_warning('Using default scaling = 0.1')
        scaling = torch.ones_like(self.xyz) * 0.1
        return scaling
    
    def init_rotation(self, num_points, device):
        rot = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
        rot[:, 0] = 1.
        return rot

    def get_rotation(self):
        self.once_warning('Using default rotation = [1, 0, 0, 0]')
        rot = torch.zeros((self.xyz.shape[0], 4), device=self.xyz.device)
        rot[:, 0] = 1.
        return rot

    @torch.no_grad()
    def prepare(self, rasterizer, camera):
        return 0
    
    def time_step(self):
        pass

    def load_state_dict(self, state_dict, strict: bool = True, silence=False, split='train'):
        # handle the shape mismatch problem
        # directly replace the tensor
        for key, val in state_dict.items():
            param = get_module_by_str(self, key)
            if param is None:
                if not silence:
                    print(f'[{self.__class__.__name__}] {key} is not in parameters, register it.')
                self.register_buffer(key, val)
                continue
            if isinstance(param, nn.Parameter):
                param.data = val
            else:
                param.set_(val)
        return True