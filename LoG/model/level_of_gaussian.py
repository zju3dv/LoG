import math
import torch
import torch.nn as nn
from .activation import Activation
from ..utils.file import create_from_point
from .sparse_optimizer import SparseOptimizer
from .counter import Counter
from .splitter import Splitter
from .tensor_tree import TensorTree
from .model_utils import get_module_by_str
from ..cuda.compute_radius import compute_radius_module
from .corrector import Corrector

MIN_PIXEL = 3

class Gaussian(nn.Module):
    def __init__(self, sh_degree=1, xyz_scale=1.) -> None:
        super().__init__()
        self.xyz_scale = xyz_scale
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.activation = Activation()
        self.keys = []

    def items(self):
        for key in self.keys:
            yield key, getattr(self, key)

    def init_rotation(self, num_points, device):
        rot = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
        rot[:, 0] = 1.
        return rot

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print(f'[{self.__class__.__name__}] one up SH degree to {self.active_sh_degree}')

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

    def init_radius3d(self, batch, renderer):
        scaling = self.activation.scaling_activation(self.scaling)
        rotation = self.activation.rotation_activation(self.rotation)
        camera, rasterizer, background = renderer.prepare_camera(batch, 0, renderer.background)
        radius2d_cuda = rasterizer.compute_radius(self.xyz, scaling, rotation)
        valid_flag = radius2d_cuda > 0
        radius3d = scaling[:, 0]
        radius3d[valid_flag] = radius3d[valid_flag] * (3./radius2d_cuda[valid_flag])
        return valid_flag, radius3d

    @torch.no_grad()
    def compute_radius(self, index, camera, level=0):
        xyz = self.xyz[index].detach()
        scaling = self.activation.scaling_activation(self.scaling[index].detach())
        rotation = self.activation.rotation_activation(self.rotation[index].detach())        
        if False:
            radius2d_cuda = compute_radius(xyz, scaling, rotation, camera)
        else: # 18.20ms
            proj_matrix = camera.raster_settings.projmatrix
            view_matrix = camera.raster_settings.viewmatrix
            tanfovx = camera.raster_settings.tanfovx
            tanfovy = camera.raster_settings.tanfovy
            image_width = camera.raster_settings.image_width
            image_height = camera.raster_settings.image_height
            focal_x = image_width / (2.0 * tanfovx)
            focal_y = image_height / (2.0 * tanfovy)
            radius2d_cuda = compute_radius_module.compute_radius(
                xyz, scaling, rotation, 
                proj_matrix, view_matrix,
                focal_x, focal_y, tanfovx, tanfovy)
        # else:
        #     radius2d_cuda = camera.compute_radius(xyz, scaling, rotation)
        scaling3d = scaling.max(dim=-1).values
        return scaling3d, radius2d_cuda

    @torch.no_grad()
    def prepare(self, rasterizer, camera):
        # 可以检查可见的node
        xyz = self.xyz.detach()
        valid_flag, depth, p_proj = self._visible_flag_by_camera(xyz, camera, padding=0.5)
        self.visibility_flag = {
            'flag': valid_flag, 
            'index': torch.where(valid_flag)[0]
        }
    
    def log_radius(self, scales):
        return f'scales: [{scales.min().item():.4f}~{scales.mean().item():.4f}~{scales.max().item():.4f}]'

class GaussianPoint(Gaussian):
    def __init__(self, init_ply, **kwargs) -> None:
        super().__init__(**kwargs)
        xyz, colors, scales = create_from_point(**init_ply)
        self.register_by_pointcloud(xyz, colors, scales, **init_ply)

    @staticmethod
    def create_from_ground(local_min, local_max, init_step, height, init_opacity=0.9, padding=0.05):
        x = torch.arange(local_min[0][0] - padding, local_max[0][0] + padding, init_step)
        y = torch.arange(local_min[0][1] - padding, local_max[0][1] + padding, init_step)
        x, y = torch.meshgrid(x, y)
        xyz = torch.stack((x, y), axis=-1).reshape(-1, 2)
        xyz = torch.cat([xyz, torch.zeros((xyz.shape[0], 1)) + height], dim=1)
        colors = torch.zeros_like(xyz) + 0.5 # 使用0.5来初始化颜色
        scaling = torch.zeros_like(xyz) + init_step
        scaling[:, 2] = init_step * 0.1 #取一个比较小的值
        opacity = torch.zeros((xyz.shape[0], 1)) + init_opacity
        return xyz, colors, scaling, opacity

    def register_by_pointcloud(self, xyz, colors, scales, init_opacity, **init_ply):
        print(f'[{self.__class__.__name__}] {self.log_radius(scales)}')
        scales = torch.clamp(scales, min=scales.mean()/4, max=scales.mean()*4)
        print(f'[{self.__class__.__name__}] -> {self.log_radius(scales)}')
        scaling = self.activation.scaling_inverse_activation(scales)[:, None].repeat(1, 3)
        colors = self.activation.rgb_inverse(colors)
        # add sh
        if self.max_sh_degree > 0:
            features = torch.zeros((colors.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32)
            shs = features
        xyz = xyz
        opacity = torch.ones_like(xyz[:, :1]) * init_opacity
        opacity = self.activation.opacity_inverse_activation(opacity)
        rotation = self.init_rotation(xyz.shape[0], xyz.device)
        # Add ground points
        if 'height' in init_ply:
            local_min, local_max = xyz.min(dim=0), xyz.max(dim=0)
            xyz_ground, colors_ground, scaling_ground, opacity_ground = self.create_from_ground(local_min, local_max, init_ply['init_step'], init_ply['height'], init_ply['ground_opacity'])
            rotation_ground = self.init_rotation(xyz_ground.shape[0], xyz.device)
            print(f'[{self.__class__.__name__}] add {xyz_ground.shape[0]} ground points')
            xyz = torch.cat([xyz, xyz_ground], dim=0)
            opacity = torch.cat([opacity, opacity_ground], dim=0)
            colors = torch.cat([colors, colors_ground], dim=0)
            scaling_ground = self.activation.scaling_inverse_activation(scaling_ground)
            scaling = torch.cat([scaling, scaling_ground], dim=0)
            rotation = torch.cat([rotation, rotation_ground], dim=0)
            if self.max_sh_degree > 0:
                shs_ground = torch.zeros((xyz_ground.shape[0], *shs.shape[1:]), dtype=torch.float32)
                shs = torch.cat([shs, shs_ground], dim=0)
        self.register_buffer('scaling', scaling)
        self.register_buffer('colors', colors)
        self.register_buffer('xyz', xyz)
        self.register_buffer('opacity', opacity)
        self.register_buffer('rotation', rotation)
        self.keys.extend(['scaling', 'colors', 'xyz', 'opacity', 'rotation'])
        if self.max_sh_degree > 0:
            self.register_buffer('shs', shs)
            self.keys.append('shs')
    
    def get_scaling_by_flag(self, index):
        scaling = self.activation.scaling_activation(self.scaling[index])
        return scaling
    
    def get_scaling_by_index_level(self, index, level):
        return self.get_scaling_by_flag(index)

    def get_rotation_by_flag(self, index):
        return self.activation.rotation_activation(self.rotation[index])

    def inverse_scaling(self, scaling, flag, depth_add=1):
        return self.activation.scaling_inverse_activation(scaling)

class LoG(nn.Module):
    def __init__(self, gaussian, tree, optimizer, densify_and_remove, use_view_correction=False):
        super().__init__()
        self.register_empty()
        self.optimizer_cfg = optimizer
        self.gaussian = GaussianPoint(**gaussian)
        self.tree = TensorTree(**tree)
        self.counter = Counter(num_points=self.gaussian.xyz.shape[0])
        self.splitter = Splitter(N=tree.max_child, split_method='uniform')
        self.densify_and_remove = densify_and_remove
        self.fix_parent = True
        self.use_view_correction = use_view_correction
        if use_view_correction:
            self.view_correction = Corrector(use_view_correction)
        self.current_depth = 0

    def __repr__(self) -> str:
        radius = self.gaussian.activation.scaling_activation(self.gaussian.scaling.detach())
        radius_max = radius.max(dim=-1).values
        opacity = self.gaussian.activation.opacity_activation(self.gaussian.opacity[:, 0].detach())
        opacity_mean = opacity.mean().item()
        return f'''Gaussian {self.num_points} points
    radius [{radius_max.min().item():.4f}~{radius_max.mean().item():.4f}~{radius_max.max().item():.4f}]
    opacity: {opacity_mean:.2f}, {(opacity<0.05).sum()} < 0.05, {(opacity<0.1).sum()} < 0.1, '''

    @property
    def num_points(self):
        return self.gaussian.xyz.shape[0]

    def register_empty(self):
        self.register_buffer('empty_xyz', torch.zeros((0, 3)))
        self.empty_xyz_parameter = nn.Parameter(torch.zeros((0, 3)))
    
    @torch.no_grad()
    def render_to_check(self, rasterizer, xyz, scaling, rotation, opacity):
        screenspace_points = torch.zeros_like(xyz, device=xyz.device)
        colors = torch.ones_like(xyz)
        ret = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = colors,
            opacities = opacity,
            scales = scaling,
            rotations = rotation,
            cov3D_precomp = None)
        rendered_image, radii, point_id_pixel, point_weight_pixel, point_weight = ret
        return point_weight
        
    def prepare(self, rasterizer, camera):
        if self.tree.num_nodes == 0:
            self.gaussian.prepare(rasterizer, camera)
        else:
            root_index = self.tree.root_index.long()
            xyz = self.gaussian.xyz[root_index].detach()
            valid_root_flag, depth, p_proj = self.gaussian._visible_flag_by_camera(xyz, camera, padding=0.5)
            # check the visiblity
            root_index_in_range = root_index[valid_root_flag]
            # render and check the weight
            opacity = self.gaussian.activation.opacity_activation(self.gaussian.opacity)
            use_visibility_check = True
            if use_visibility_check:
                scaling = self.gaussian.activation.scaling_activation(self.gaussian.scaling)
                rotation = self.gaussian.activation.rotation_activation(self.gaussian.rotation)
                point_weight = self.render_to_check(rasterizer,
                    xyz[root_index_in_range], scaling[root_index_in_range],
                    rotation[root_index_in_range], opacity[root_index_in_range])
                valid_root_flag[valid_root_flag.clone()] = point_weight > 1e-8
            root_index = root_index[valid_root_flag]
            index_all = self.tree.traverse(self.gaussian, root_index, rasterizer, max_depth=self.current_depth)
            if self.optimizer_cfg.opt_all_levels:
                # optimize the leaf nodes in all levels
                flag_isleaf = (self.tree.node_index[index_all] == -1) & (self.tree.depth[index_all] > 0)
            else:
                # only optimize the leaf nodes in last level
                flag_isleaf = self.tree.depth[index_all] == self.current_depth
            index_leaf = index_all[flag_isleaf]
            index_node = index_all[~flag_isleaf]
            self.gaussian.visibility_flag = {
                'root_flag': valid_root_flag, 
                'index': index_leaf,
                'index_node': index_node
            }

    @property
    def visibility_flag(self):
        return self.gaussian.visibility_flag

    def get_all(self, camera, rasterizer):
        visible_index = self.gaussian.visibility_flag['index']
        # This design optimize the node either
        if self.fix_parent:
            ret = {}
            for key, val in self.gaussian.items():
                if self.training:
                    ret[key] = nn.Parameter(val[visible_index])
                else:
                    ret[key] = val[visible_index]
            self.gaussian.visibility_flag['params'] = ret
            ret_full = {}
            if 'index_node' in self.gaussian.visibility_flag.keys():
                index_node = self.gaussian.visibility_flag['index_node']
                for key, val in self.gaussian.items():
                    ret_full[key] = torch.cat([ret[key], val[index_node]])
            else:
                ret_full = ret
            ret = ret_full
        else:
            if 'index_node' in self.gaussian.visibility_flag.keys():
                visible_index = torch.cat([
                    visible_index, 
                    self.gaussian.visibility_flag['index_node']])
            ret = {}
            for key, val in self.gaussian.items():
                if self.training:
                    ret[key] = nn.Parameter(val[visible_index])
                else:
                    ret[key] = val[visible_index]
            self.gaussian.visibility_flag['params'] = ret
        ret = self.gaussian.activation.activate_root_return(
            ret, camera, self.gaussian.active_sh_degree)
        ret['scaling'] = ret['scaling']
        return ret

    # training state
    def set_stage(self, stage_name):
        self.stage_name = stage_name

    def set_state(self, active_sh_degree=None, enable_sh=None,
                  min_resolution_pixel=None, current_depth=None,
                  scaling_modifier=1.,
                  log_query=None,
                  reset_created_steps=False):
        if active_sh_degree is not None or enable_sh is not None:
            if enable_sh:
                self.gaussian.active_sh_degree = self.gaussian.max_sh_degree
            else:
                self.gaussian.active_sh_degree = min(active_sh_degree, self.gaussian.max_sh_degree)
            print(f'[{self.__class__.__name__}] active_sh_degree: {self.gaussian.active_sh_degree}')
        if reset_created_steps:
            self.gaussian.create_steps.fill_(0.)
            print(f'[{self.__class__.__name__}] reset created steps')
        if min_resolution_pixel is not None:
            self.tree.min_resolution_pixel = min_resolution_pixel.item()
        if current_depth is not None:
            self.current_depth = current_depth.item()
            print(f'[{self.__class__.__name__}] set current depth -> {self.current_depth}')
        if log_query is not None:
            self.tree.log_query = log_query
    
    # initialize
    def at_init_start(self):
        self.num_views = 0

    def init(self, renderer, batch, iteration):
        # use projection to get the scaling mean
        valid_flag, r3d = self.gaussian.init_radius3d(batch, renderer)
        self.counter.radius3d_min[valid_flag] = torch.minimum(self.counter.radius3d_min[valid_flag], r3d[valid_flag])
        self.num_views += 1

    def at_init_final(self):
        print(f'[{self.__class__.__name__}] minimum {self.gaussian.log_radius(self.counter.radius3d_min)}')
        radius3d_min = self.gaussian.activation.scaling_inverse_activation(self.counter.radius3d_min)[:, None].repeat(1, 3)
        # update radius3d_max
        self.counter.radius3d_max.fill_(self.gaussian.xyz_scale * 0.2)
        self.gaussian.scaling = torch.maximum(self.gaussian.scaling, radius3d_min)
        if self.use_view_correction:
            self.view_correction.init(self.num_views)
    
    # training setup
    def training_setup(self):
        cfg = self.optimizer_cfg
        if getattr(self, 'optimizer', None) is not None:
            print(f'[{self.__class__.__name__}] optimizer is already setup')
            print(f'[{self.__class__.__name__}] optimizer device: {self.optimizer.exp_avg.device}')
            self.counter.reset(self.num_points)
            return 0
        cfg.lr_dict['max_steps'] *= self.base_iter
        self.optimizer = SparseOptimizer(cfg.optimize_keys, cfg.lr_dict, self.gaussian, xyz_scale=self.gaussian.xyz_scale)
        print(f'[{self.__class__.__name__}] optimizer setup: max steps = {cfg.lr_dict["max_steps"]}')
 
        self.lr = cfg.lr_dict['xyz']
        self.counter.reset(self.num_points)
        if self.use_view_correction:
            self.view_correction.training_setup()

    # training step
    def clear(self):
        self.gaussian.visibility_flag = None

    def update_by_output(self, output):
        self.counter.update_by_output(output, fix_parent=self.fix_parent)

    def clamp_scale(self, index):
        apply_scaling_clip = True
        if apply_scaling_clip:
            scaling = self.gaussian.scaling[index]
            # scaling_max = 0.2 * self.gaussian.xyz_scale + torch.zeros_like(scaling)
            scaling_max = self.counter.radius3d_max[index][:, None].expand(-1, 3)
            scaling_min = self.counter.radius3d_min[index][:, None].expand(-1, 3)
            scaling = torch.clamp(scaling, 
                    self.gaussian.activation.scaling_inverse_activation(scaling_min), 
                    self.gaussian.activation.scaling_inverse_activation(scaling_max))
            self.gaussian.scaling[index] = scaling

    def step(self):
        params = self.visibility_flag['params']
        index = self.visibility_flag['index']
        flag_vis = self.visibility_flag['flag_vis']
        if 'index_node' in self.visibility_flag.keys() and self.visibility_flag['index_node'].shape[0] > 0:
            if self.fix_parent:
                flag_vis = flag_vis[:index.shape[0]]
            else:
                index = torch.cat([
                    index, 
                    self.visibility_flag['index_node']])
        self.optimizer.step(self.gaussian, index, params, flag_vis)
        # clip the scaling
        index = index[flag_vis]
        self.clamp_scale(index)
        self.lr = self.optimizer.xyz_lr
        if self.optimizer.global_steps == self.base_iter:
            print(f'[{self.__class__.__name__}] base iteration {self.base_iter} done, enable view_correction module')
        if self.use_view_correction and self.optimizer.global_steps > self.base_iter:
            self.view_correction.step()

    def update_init_stage(self, scale=1):
        flag_remove_weight = self.counter.weights_max < self.densify_and_remove.init_weight_min
        opacity = self.gaussian.activation.opacity_activation(self.gaussian.opacity[:, 0].detach())
        flag_nonmax = self.counter.weights_max < opacity * 0.1
        flag_remove_small = self.counter.radii_max_max < (self.densify_and_remove.init_radius_min * scale) ** 2
        print(f'[{self.__class__.__name__}] {flag_remove_weight.sum():10d} points with weight < {self.densify_and_remove.init_weight_min:.2f}')
        print(f'[{self.__class__.__name__}] {flag_nonmax.sum():10d} points with weight is non max')
        print(f'[{self.__class__.__name__}] {flag_remove_small.sum():10d} points with radius < {self.densify_and_remove.init_radius_min:.2f}')
        flag_remove_small = flag_remove_small & (torch.rand_like(self.counter.weights_max) > 0.5)
        flag_remove = flag_remove_small | flag_remove_weight | flag_nonmax
        radii_max = self.counter.radii_max_max.float()
        flag_activation = (self.counter.create_steps > self.densify_and_remove.min_steps) & (radii_max > 0)
        print(f'[{self.__class__.__name__}] {self.counter.str_min_mean_max("radii_max_act", radii_max[flag_activation])}')        
        grad = self.counter.get_gradmean()
        print(f'[{self.__class__.__name__}] {self.counter.str_min_mean_max("grad", grad)}')        
        radii_mean = radii_max[flag_activation].mean()
        radii_std = radii_max[flag_activation].std()
        mode = self.densify_and_remove.init_split_method
        split_thres = self.densify_and_remove.init_radius_split * scale
        if mode == 'split_by_2d':
            if split_thres == -1:
                split_thres = radii_mean + radii_std * 3
            flag_split_grad = (grad > 10 * self.densify_and_remove.split_grad_thres) & (radii_max > self.densify_and_remove.init_radius_min * scale * 8)
            flag_split_radii = radii_max > split_thres ** 2
            print(f'[{self.__class__.__name__}] split by grad : {flag_split_grad.sum():8d}')
            print(f'[{self.__class__.__name__}] split by radii: {flag_split_radii.sum():8d}')
            flag_split = flag_split_radii | flag_split_grad
            flag_split = flag_activation & flag_split & (~flag_remove)
            print(f'[{self.__class__.__name__}] {self.counter.str_min_mean_max("radii_split", radii_max[flag_split])}')
        elif mode == 'split_by_3d':
            radius = self.gaussian.activation.scaling_activation(self.gaussian.scaling.detach())
            radius_max = radius.max(dim=-1).values
            # flag_split = radii_max > radii_max.mean() + 3 * radii_std
            flag_split  = radius_max > self.gaussian.xyz_scale * 0.1
            flag_remove2d = radius_max < self.gaussian.xyz_scale * 0.005
            print(f'[{self.__class__.__name__}] radii: {radii_mean:.2f}+{radii_std:.2f}')
            print(f'[{self.__class__.__name__}] large {flag_split.sum():10d} points > {self.gaussian.xyz_scale*0.1:.4f}')
            print(f'[{self.__class__.__name__}] small {flag_remove2d.sum():10d} points < {self.gaussian.xyz_scale*0.005:.4f}')
            flag_remove2d = flag_activation & flag_remove2d
            # random dropout
            flag_rand = torch.rand_like(radii_max) > 0.5
            flag_remove = (flag_remove2d & flag_rand) | flag_remove
            self.counter.create_steps[(flag_remove2d & (~flag_rand))] = 0
            flag_split = flag_split & (~flag_remove)
        self.splitter.split_and_remove(self.gaussian, self.optimizer, flag_split, flag_remove)
        self.splitter.split_and_remove_other(self.counter, ['create_steps', 'radius3d_min', 'radius3d_max'], flag_split, flag_remove)
        self.counter.radius3d_max.fill_(0.2*self.gaussian.xyz_scale)
        index = torch.arange(0, self.num_points, device=self.gaussian.xyz.device)
        self.clamp_scale(index)
        radius = self.gaussian.activation.scaling_activation(self.gaussian.scaling.detach())
        radius_max = radius.max(dim=-1).values
        print(f'[{self.__class__.__name__}] {self.counter.str_min_mean_max("radius3d_min", self.counter.radius3d_min)}')
        self.counter.reset(self.num_points)

    def update_depth_stage(self, global_iteration):
        log_prefix = f'[{self.__class__.__name__}] {global_iteration:06d}'
        opacity = self.gaussian.activation.opacity_activation(self.gaussian.opacity[:, 0].detach())
        radius = self.gaussian.activation.scaling_activation(self.gaussian.scaling.detach())
        radius_max = radius.max(dim=-1).values
        radius_min = radius.min(dim=-1).values
        radius_mid = radius.sum(dim=-1) - radius_max - radius_min
        ratio = radius_max / radius_mid
        # 
        only_operate_last_layer = False
        if only_operate_last_layer:
            flag_is_parent = self.tree.depth == self.current_depth - 1
            depth_minus1_sum = (self.tree.depth == self.current_depth - 1).sum()
            flag_depth_parent = (self.tree.depth == self.current_depth - 1) & (self.tree.node_index == -1)
            flag_depth_child = self.tree.depth == self.current_depth
        else:
            flag_is_parent = (self.tree.node_index == -1) & (self.tree.depth < self.current_depth)
            flag_depth_parent = flag_is_parent & (self.counter.create_steps > self.densify_and_remove.min_steps_split)
            depth_minus1_sum = (self.tree.depth < self.current_depth).sum()
            flag_depth_child = (self.tree.node_index == -1) & (self.tree.depth > 0)
        grad = self.counter.get_gradmean()
        radii_max_max = self.counter.radii_max_max.float()
        print(f'{log_prefix} {self.counter.str_min_mean_max("opacity", opacity[flag_is_parent])}')
        print(f'{log_prefix} {self.counter.str_min_mean_max("ratio", ratio[flag_is_parent])}')
        print(f'{log_prefix} {self.counter.str_min_mean_max("grad", grad[flag_is_parent])}')
        print(f'{log_prefix} {self.counter.str_min_mean_max("radii", radii_max_max[flag_is_parent])}')
        grad_thres = self.densify_and_remove.split_grad_thres
        radius2d_thres = self.densify_and_remove.radius2d_thres
        flag_split_grad = grad > grad_thres
        flag_split_radii = self.counter.radii_max_max > radius2d_thres
        print(f'{log_prefix} split by grad: {flag_split_grad.sum():8d} split by radii: {flag_split_radii.sum():8d}')
        flag_split = flag_split_grad & flag_split_radii & flag_depth_parent
        if flag_depth_child.sum() == 0: # not any children
            flag_remove = torch.zeros_like(flag_split)
        else:
            weights_max = self.counter.weights_max
            flag_remove = flag_depth_child & (weights_max < self.densify_and_remove.remove_weights_thres) & (self.counter.visible_count > 1)
        flag_split = flag_split & (~flag_remove)
        num_max_split = min(int(depth_minus1_sum * 0.05), self.densify_and_remove.max_split_points)
        sort_method = self.densify_and_remove.sort_method
        if flag_split.sum() > num_max_split:
            if sort_method == 'radii':
                radii_max_max_depth = radii_max_max[flag_split]
                new_radii_thres = torch.topk(radii_max_max_depth, num_max_split, largest=True).values[-1]
                print(f'{log_prefix} select top {num_max_split} points to split. New radii thres = {new_radii_thres:.1f}')
                flag_split = flag_split & (radii_max_max >= new_radii_thres)
            elif sort_method == 'opacity':
                opacity = self.gaussian.activation.opacity_activation(self.gaussian.opacity[:, 0].detach())
                new_opacity_thres = torch.topk(opacity[flag_split], num_max_split, largest=True).values[-1]
                print(f'{log_prefix} select top {num_max_split} points to split. New opacity thres = {new_opacity_thres:.3f}')
                flag_split = flag_split & (opacity >= new_opacity_thres)
            elif sort_method == 'grad':
                new_grad_thres = torch.topk(grad[flag_split], num_max_split, largest=True).values[-1]
                print(f'{log_prefix} select top {num_max_split} points to split. New grad thres = {new_grad_thres:.3f}')
                flag_split = flag_split & (grad >= new_opacity_thres)
        flag_split, flag_remove = self.tree.split_and_remove(flag_split, flag_remove)
        self.splitter.split_and_remove(self.gaussian, self.optimizer, 
            flag_split, flag_remove, remove_split=False)
        # update the tree
        self.splitter.split_and_remove_other(self.counter, ['create_steps', 'radius3d_min', 'radius3d_max'], 
            flag_split, flag_remove, remove_split=False)
        # set the radius3d_max to the parent scales
        num_split = flag_split.sum() * self.splitter.N
        scaling_decay = self.densify_and_remove.scaling_decay
        if num_split > 0:
            self.counter.radius3d_max[-num_split:] = scaling_decay * radius_max[flag_split][:, None].repeat(1, self.splitter.N).reshape(-1)
        self.counter.reset(self.num_points)
        for depth in range(self.current_depth + 1):
            flag_depth = self.tree.depth == depth
            if flag_depth.sum() == 0:
                continue
            print(f'[{self.__class__.__name__}] depth = {depth:2d} | {flag_depth.sum():10d} points')

    def upgrade_tree(self):
        if self.current_depth == 0:
            self.tree.initialize(self.gaussian.xyz)
        # self.current_depth += 1
        self.current_depth = 20
        print(f'[{self.__class__.__name__}] current depth: {self.current_depth}')
        self.counter.reset(self.num_points)

    def update_by_iteration(self, iteration, global_iteration):
        base_iter = self.base_iter
        # update sh
        upgrade_sh_iter = self.densify_and_remove.upgrade_sh_iter * base_iter
        if global_iteration > 0 and (global_iteration + 1) % upgrade_sh_iter == 0:
            self.gaussian.oneupSHdegree()
        densify_from_iter = self.densify_and_remove.densify_from_iter * base_iter
        densify_every_iter = self.densify_and_remove.densify_every_iter * base_iter
        sum_iter = int((self.current_depth + 1)*(self.current_depth+2)/2)
        upgrade_tree_iter = densify_every_iter * sum_iter * 20
        sum_iter = (self.current_depth + 1)
        upgrade_tree_iter = densify_every_iter * sum_iter * self.densify_and_remove.upgrade_repeat
        if (iteration + 1) == densify_from_iter:
            self.counter.reset(self.num_points)
            return False
        if (iteration + 1 > densify_from_iter) and (iteration + 1) % densify_every_iter == 0:
            if (iteration + 1) % upgrade_tree_iter == 0 and self.stage_name != 'init':
                self.upgrade_tree()
                return True
            if self.current_depth == 0:
                if self.stage_name == 'init':
                    self.update_init_stage()
                else:
                    self.update_init_stage(scale=2)
            else:
                if (iteration + 1) % (2*densify_every_iter) == 0:
                    self.update_depth_stage(global_iteration)
                else:
                    self.counter.reset(self.num_points)
            return True
        return False

    def load_state_dict(self, state_dict, strict= True, split='demo'):
        # handle the shape mismatch problem
        if split == 'train':
            self.training_setup()
        # directly replace the tensor
        for key, val in state_dict.items():
            if split != 'train' and 'optimizer' in key:
                print(f'Skip the {key} as split = {split}')
                continue
            param = get_module_by_str(self, key)
            if param is None:
                print(f'[{self.__class__.__name__}] {key} is not in parameters, register it.')
                self.register_buffer(key, val)
                continue
            if param.device != val.device:
                val = val.to(param.device)
            if param.dtype != val.dtype:
                print(key, param.shape, param.dtype, val.shape, val.dtype)
                val = val.to(param.dtype)
            if isinstance(param, nn.Parameter):
                param.data = val
            else:
                param.set_(val)
        if self.tree.num_nodes > 0:
            self.current_depth = self.tree.depth.max().item()
        return True