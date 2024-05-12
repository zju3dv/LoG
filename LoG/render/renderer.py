from diff_gaussian_rasterization_wodilate import GaussianRasterizationSettings, GaussianRasterizer
import math
import os
import time
import numpy as np
from collections import defaultdict
import cv2
import torch
import torch.nn as nn

class BaseRender(torch.nn.Module):
    GaussianRasterizationSettings = GaussianRasterizationSettings
    GaussianRasterizer = GaussianRasterizer
    @staticmethod
    def float32_to_uint8(array):
        return np.clip(array*255, 0, 255).astype(np.uint8)

    @staticmethod
    def tensor_to_bgr(tensor):
        vis = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        vis = (np.clip(vis[:,:,::-1], 0., 1.)*255).astype(np.uint8)
        vis = np.ascontiguousarray(vis)
        return vis
    
    @staticmethod
    def make_video(path, remove_image=False, fps=30):
        cmd = f'ffmpeg -y -r {fps} -i {path}/%06d.jpg  -vf scale="2*ceil(iw/2):2*ceil(ih/2)" -vcodec libx264 -r {fps} {path}.mp4 -loglevel quiet'
        print(cmd)
        os.system(cmd)

    @staticmethod
    def acc_to_bgr(tensor):
        vis = tensor.detach().cpu().numpy()
        vis = (np.clip(vis, 0., 1.)*255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        vis = np.ascontiguousarray(vis)
        return vis
    
    @staticmethod
    def depth_to_bgr(tensor):
        tensor = tensor.detach()
        depth = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return BaseRender.acc_to_bgr(depth)    
    
    @staticmethod
    def marigold_depth_vis(tensor, cmap="Spectral"):
        import matplotlib
        if torch.is_tensor(tensor):
            depth = tensor.detach().cpu().numpy()
        else:
            depth = tensor
        cm = matplotlib.colormaps[cmap]
        img_colored_np = cm(depth, bytes=False)[..., 0:3]  # value from 0 to 1
        return BaseRender.float32_to_uint8(img_colored_np)

    @staticmethod
    def prepare(viewpoint_camera, background, scaling_modifier=1.):
        if not isinstance(viewpoint_camera, dict):
            viewpoint_camera = viewpoint_camera.to_dict()
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera['FoVx'] * 0.5)
        tanfovy = math.tan(viewpoint_camera['FoVy'] * 0.5)
        raster_settings = BaseRender.GaussianRasterizationSettings(
            image_height=int(viewpoint_camera['image_height']),
            image_width=int(viewpoint_camera['image_width']),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera['world_view_transform'],
            projmatrix=viewpoint_camera['full_proj_transform'],
            sh_degree=0,
            campos=viewpoint_camera['camera_center'],
            prefiltered=False,
            debug=False
        )
        rasterizer = BaseRender.GaussianRasterizer(raster_settings=raster_settings)
        return rasterizer

class NaiveRendererAndLoss(BaseRender):
    def __init__(self, split='train', use_randback=False, background=[0., 0., 0.], 
                 use_rand_radius=False,
                 use_origin_render=False,
                 render_depth=False,
                 ):
        super().__init__()
        self.split = split
        self.use_randback = use_randback
        self.use_rand_radius = use_rand_radius
        self.render_depth = render_depth
        if render_depth:
            from .loss import ScaleAndShiftInvariantLoss
            self.depth_loss = ScaleAndShiftInvariantLoss()
        background = torch.tensor(background, dtype=torch.float32)
        self.register_buffer('background', background)
        self.l1_loss = nn.L1Loss()
        from .loss import SSIM
        self.ssim_loss = SSIM(11, 3)
        if use_origin_render:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
            BaseRender.GaussianRasterizationSettings = GaussianRasterizationSettings
            BaseRender.GaussianRasterizer = GaussianRasterizer
        else:
            from diff_gaussian_rasterization_wodilate import GaussianRasterizationSettings, GaussianRasterizer
            BaseRender.GaussianRasterizationSettings = GaussianRasterizationSettings
            BaseRender.GaussianRasterizer = GaussianRasterizer
        self.use_origin_render = use_origin_render

    def set_state(self, render_depth=None, background=None):
        if render_depth is not None:
            self.render_depth = render_depth
        if background is not None:
            print(f'[{self.__class__.__name__}] Set background to {background}')
            background = torch.tensor(background, dtype=torch.float32, device=self.background.device)
            self.background = background

    def render(self, camera, rasterizer, model, extra_params={}):
        ret = model.get_all(camera, rasterizer, **extra_params)
        if len(ret) == 0:
            device = camera['world_view_transform'].device
            ret = {
                'xyz': torch.zeros((0, 3), device=device),
                'opacity': torch.zeros((0,), device=device),
                'colors': torch.zeros((0, 3), device=device),
                'scaling': torch.zeros((0, 3), device=device) + 0.1,
                'rotation': torch.zeros((0, 4), device=device)
            }
        xyz = ret['xyz']
        opacity = ret['opacity']
        colors = ret['colors']
        scales = ret['scaling']
        rotations = ret['rotation']
        cov3D = None
        empty_xyz = model.empty_xyz
        screenspace_points = torch.zeros_like(xyz, dtype=empty_xyz.dtype, device=empty_xyz.device, requires_grad=True)
        try:
            screenspace_points.retain_grad()
        except:
            pass
        model_data = ret
        name_args = {
            'means3D': xyz,
            'means2D': screenspace_points,
            'shs': None,
            'colors_precomp': colors,
            'opacities': opacity,
            'scales': scales,
            'rotations': rotations,
            'cov3D_precomp': cov3D
        }
        if not self.use_origin_render and not model.training:
            name_args['use_filter'] = False
        ret = rasterizer(**name_args)
        if len(ret) == 5:
            rendered_image, radii, point_id_pixel, point_weight_pixel, point_weight = ret
            point_id, point_count = torch.unique(point_id_pixel, sorted=True, return_counts=True)
            if point_id[0] == -1:
                point_id = point_id[1:]
                point_count = point_count[1:]
        else:
            rendered_image, radii = ret
            point_weight_pixel = None
            point_id = torch.zeros_like(xyz[:, 0], dtype=torch.int32)
            point_count = torch.zeros_like(xyz[:, 0], dtype=torch.int32)
            point_weight = torch.zeros_like(xyz[:, 0], dtype=torch.float32)
        render_time = 0
        xyz1 = torch.cat([xyz.detach(), torch.ones_like(xyz[:, :1])], dim=1)
        xyz1_RT = xyz1 @ camera['world_view_transform']
        point_depth = xyz1_RT[:, 2]
        ret = {
            "render": rendered_image,
            "point_id": point_id,
            "point_count": point_count,
            "point_weight": point_weight,
            "focal": max(camera['K'][0, 0], camera['K'][1, 1]),
            "point_depth": point_depth,
            "viewspace_points": screenspace_points,
            "radii": radii,
            "visibility_flag": model.visibility_flag,
            "xyz": xyz,
            "colors": colors,
            "scales": scales, # scales of visibility points
            "opacity": opacity,
            "render_time": render_time
        }
        if self.render_depth:
            ones = torch.ones_like(point_depth)
            height = xyz[:, 2]
            colors_depth = torch.stack([point_depth, height, ones], dim=-1)
            ret_depth = rasterizer(
                means3D = xyz,
                means2D = screenspace_points,
                shs = None,
                colors_precomp = colors_depth,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D)
            ret['depth'] = ret_depth[0][0]
            ret['height'] = ret_depth[0][1]
            ret['accmap'] = ret_depth[0][2]
        if point_weight_pixel is not None:
            ret["point_weight_pixel"] = point_weight_pixel.data
        ret['render'] = rendered_image[:3]
        return ret, model_data

    def prepare_camera(self, batch, bn, background, is_train=False):
        camera = {}
        for key in ['camera_center', 'world_view_transform', 'full_proj_transform', 'image_width', 'image_height', 'FoVx', 'FoVy', 'K', 'R', 'T']:
            camera[key] = batch['camera'][key][bn]
        for key in ['nframe', 'nview']:
            if key in batch:
                camera[key] = batch[key][bn]
        if background is not None:
            if not torch.is_tensor(background):
                background = torch.tensor(background).to(self.background.device)
        else:
            if is_train and self.use_randback:
                background = torch.rand_like(self.background)
            else:
                background = self.background
        rasterizer = self.prepare(camera, background, scaling_modifier=1)
        return camera, rasterizer, background

    def vis(self, batch, model, background=None):
        preds = defaultdict(list)
        for bn in range(batch['camera']['camera_center'].shape[0]):
            camera, rasterizer, background = self.prepare_camera(batch, bn, background, is_train=model.training)
            if model.training and self.use_rand_radius:
                origin_radius = model.tree.min_resolution_pixel
                random_log2 = torch.rand(1).item()
                if random_log2 > 0.5:
                    # random_log2: (0.5, 1) => (1, 5) => (2, 32)
                    pixel_radius = 3 * 2 ** (random_log2 * 8 - 3)
                else:
                    # random_log2: (0, 0.5) => (0, 1) => (1, 2)
                    pixel_radius = 3 * 2 ** (random_log2 * 2)
                model.tree.min_resolution_pixel = pixel_radius
            model.prepare(rasterizer, camera)
            render_pkg, model_data = self.render(camera, rasterizer, model)
            if model.training and self.use_rand_radius:
                model.tree.min_resolution_pixel = origin_radius
            if getattr(model, 'view_correction', None) is not None and model.training:
                view_correction = model.view_correction[batch['index'][bn].item()]
                render_pkg['render_correct'] = render_pkg['render'] * view_correction[:, None, None]
            for key, val in render_pkg.items():
                preds[key].append(val)
        for key in ['render', 'render_correct', 'render_max']:
            if key in preds.keys():
                preds[key] = torch.stack(preds[key])
        return preds

    def calculate_loss(self, gt_image, render, output, mask_ignore=None):
        if mask_ignore is not None:
            render = gt_image * mask_ignore[:, None] + render * (1 - mask_ignore[:, None])
        ssim_loss = self.ssim_loss(render, gt_image)
        if 'render_correct' in output.keys():
            render_l1 = output['render_correct'][:, :3]
        else:
            render_l1 = render
        l1_loss = self.l1_loss(render_l1, gt_image)
        output['loss_dict'] = {
            'l1': l1_loss.item(),
            'ssim': ssim_loss.item()
        }
        output['loss'] = 0.2*ssim_loss + 0.8*l1_loss
    
    def append_depth_loss(self, gt_depth, pred_depth, output):
        accmap = output['accmap'][0]
        mask = accmap > 0.5
        gt_depth = gt_depth[0]
        pred_depth = pred_depth[0]
        num_patch = 64
        patch_size = 64
        preds, gts, masks = [], [], []
        start_rows = torch.randint(0, gt_depth.shape[0] - patch_size, size=(num_patch,), device=gt_depth.device)
        start_cols = torch.randint(0, gt_depth.shape[1] - patch_size, size=(num_patch,), device=gt_depth.device)
        for i in range(num_patch):
            preds.append(pred_depth[start_rows[i]:start_rows[i]+patch_size, start_cols[i]:start_cols[i]+patch_size])
            gts.append(gt_depth[start_rows[i]:start_rows[i]+patch_size, start_cols[i]:start_cols[i]+patch_size])
            masks.append(mask[start_rows[i]:start_rows[i]+patch_size, start_cols[i]:start_cols[i]+patch_size])
        preds = torch.stack(preds)
        gts = torch.stack(gts)
        masks = torch.stack(masks)
        depth_loss, _ = self.depth_loss(1./(preds + 1e-5), gts, mask=masks)
        output['gt_depth'] = gt_depth[None]
        pred_depth_vis = 1./(pred_depth.detach() + 1e-5)
        pred_depth_vis = (pred_depth_vis - pred_depth_vis[mask].min())/(pred_depth_vis[mask].max() - pred_depth_vis[mask].min())
        output['pred_depth'] = pred_depth_vis[None]
        output['loss_dict']['depth'] = depth_loss
        output['loss'] += 1. * depth_loss
        return output

    def process_gt(self, batch):
        return batch['image'].permute(0, 3, 1, 2)

    def process_pred(self, batch, pred):
        return pred

    def forward(self, batch, model):
        output = self.vis(batch, model)
        if self.split == 'train' or self.split == 'val':
            gt_image = batch['image'].permute(0, 3, 1, 2)
            render = output['render'][:, :3]
            if self.split == 'train':
                if 'mask_ignore' in batch.keys():
                    mask_ignore = batch['mask_ignore']
                    gt_image = gt_image * mask_ignore + (1 - mask_ignore) * output['background'][0][:, None, None]
                    self.calculate_loss(gt_image, render, output, mask_ignore=mask_ignore)
                    output['mask_ignore'] = mask_ignore
                else:
                    self.calculate_loss(gt_image, render, output)
                if self.render_depth:
                    self.append_depth_loss(batch['depth'], output['depth'], output)
            output['gt'] = gt_image
        return output

class MaskForeground(NaiveRendererAndLoss):
    def bound_from_mask(self, msk, padding):
        assert msk.shape[0] == 1, 'only support batch size 1'
        msk_hw = msk[0, :, :, 0] > 0.5
        msk_hw_0 = torch.where(msk_hw.any(dim=0))[0]
        msk_hw_1 = torch.where(msk_hw.any(dim=1))[0]
        l, r = max(msk_hw_0[0] - padding, 0), msk_hw_0[-1] + padding
        t, b = max(msk_hw_1[0] - padding, 0), msk_hw_1[-1] + padding
        return l, t, r, b
    
    def process_gt(self, batch):
        msk = batch['mask'][..., None]
        l, t, r, b = self.bound_from_mask(msk, padding=0)
        gt = batch['image']
        msk = batch['mask'][..., None]
        gt = gt * msk + (1 - msk) * self.background[None, None, None]
        gt = gt[:, t:b+1, l:r+1]
        return gt.permute(0, 3, 1, 2)

    def process_pred(self, batch, pred):
        msk = batch['mask'][..., None]
        l, t, r, b = self.bound_from_mask(msk, padding=0)
        pred = pred[:, t:b+1, l:r+1]
        return pred

    def forward(self, batch, model):
        if self.split == 'train':
            gt = batch['image']
            if self.use_randback:
                rand_bkgd = torch.rand(3, device=gt.device)
            else:
                rand_bkgd = self.background
            msk = batch['mask'][..., None]
            # crop the region
            padding = int(max(msk.shape)/50)
            l, t, r, b = self.bound_from_mask(msk, padding)
            gt = gt[:, t:b+1, l:r+1]
            msk = msk[:, t:b+1, l:r+1]
            gt = gt * msk + (1 - msk) * rand_bkgd[None, None, None]
            output = self.vis(batch, model, background=rand_bkgd)
            gt_image = gt.permute(0, 3, 1, 2)
            render = output['render'][:, :3]
            render = render[:, :, t:b+1, l:r+1]
            # get the region of interest
            if 'mask_ignore' in batch.keys():
                mask_ignore = batch['mask_ignore']
                mask_ignore = mask_ignore[:, t:b+1, l:r+1]
                self.calculate_loss(gt_image, render, output, mask_ignore=mask_ignore)
                output['mask_ignore'] = mask_ignore
            else:
                self.calculate_loss(gt_image, render, output)
            output['gt'] = gt_image
            output['render'] = render
        else:
            output = self.vis(batch, model)
        return output