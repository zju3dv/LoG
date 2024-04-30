import torch
from .sh_utils import RGB2SH, SH2RGB

class Activation:
    def __init__(self, scaling_activation='exp'):
        if scaling_activation == 'exp':
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        elif scaling_activation == 'sigmoid':
            self.scaling_activation = torch.sigmoid
            self.scaling_inverse_activation = lambda x: torch.log((x)/(1-x))
        elif scaling_activation == 'tanh':
            self.scaling_activation = torch.tanh
            self.scaling_inverse_activation = torch.arctanh
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = lambda x: torch.log((x)/(1-x))
        self.rotation_activation = torch.nn.functional.normalize
        from .sh_utils import eval_sh_wobase
        self.sh_activation = eval_sh_wobase
        self.rgb_inverse = RGB2SH

    def init_rotation(self, num_points, device):
        rot = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
        rot[:, 0] = 1.
        return rot

    def colors_activation(self, ret, camera, active_sh_degree):
        colors = SH2RGB(ret['colors'])
        if active_sh_degree > 0 and camera is not None:
            xyz = ret['xyz'].detach()
            dir_pp = xyz - camera['camera_center'][None]
            dir_pp = dir_pp / torch.norm(dir_pp, dim=-1, keepdim=True)
            colors = colors + self.sh_activation(dir_pp, ret['shs'], degree=active_sh_degree)
        return colors

    def activate_root_return(self, ret, camera, active_sh_degree):
        ret_new = {
            'xyz': ret['xyz'],
            'scaling': self.scaling_activation(ret['scaling']),
            'opacity': self.opacity_activation(ret['opacity']),
            'rotation': self.rotation_activation(ret['rotation']),
            'colors': self.colors_activation(ret, camera, active_sh_degree)
        }
        return ret_new