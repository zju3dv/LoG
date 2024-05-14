import numpy as np
import cv2
from .base import prepare_camera
from .demo import DemoBase

class OverlookByScale(DemoBase):
    def __init__(self, 
        focal, shape, ground_height,
        rotate_x=0, lookat=[0, 0, 0],
        step=100, scales=[1, 2], border_length=1,
        axis_up='z', znear=0.01, zfar=100) -> None:
        super().__init__(znear=znear, zfar=zfar)
        lookat[2] += ground_height
        width, height = shape
        K = np.array([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        if axis_up == 'z':
            R = np.array([
                [1.,  0.,  0.],
                [0., 1., 0.],
                [0., 0.,  1.]], dtype=np.float32)
        elif axis_up == '-z':
            R = np.array([
                [1.,  0.,  0.],
                [0., -1., 0.],
                [0., 0.,  -1.]], dtype=np.float32)
        if False:
            scale_space = np.linspace(scales[0], scales[1], step)
            dist = focal / (scale_space * max(width, height)) * border_length
        else:
            scale_space = np.logspace(np.log10(scales[0]), np.log10(scales[1]), step)
            dist = focal / (scale_space * min(width, height)) * border_length
        
        if axis_up == 'z':
            z = (- dist) * np.cos(np.deg2rad(rotate_x)) + lookat[2]
        elif axis_up == '-z':
            z = (  dist) * np.cos(np.deg2rad(rotate_x)) + lookat[2]
        y = (- dist) * np.sin(np.deg2rad(rotate_x)) + lookat[1]
        x = np.zeros_like(z) + lookat[0]
        center = np.stack([x, y, z], axis=-1)
        # rotate the camera
        Rrel = np.deg2rad(np.array([[rotate_x, 0., 0.]]))
        Rrel = cv2.Rodrigues(Rrel)[0]
        R = R @ Rrel
        # 
        infos = []
        for center_ in center:
            center_ = np.array(center_).reshape(3, 1)
            T = - R @ center_
            camera={
                'K': K, 'R': R, 'T': T,
                'H': height, 'W': width,
                'center': center_
            }
            infos.append({
                'camera': camera,
                'scale': 1
            })
        self.infos = infos
        # write_cameras_all(self.camera_vis_all)

class LookAt(DemoBase):
    def __init__(self, K, H, W, scale,
        lookat, radius, angle, znear=0.1, zfar=100., ranges=[0, 360, 181]) -> None:
        super().__init__(znear=znear, zfar=zfar)
        self.K = np.array(K, dtype=np.float32)
        self.H = H
        self.W = W
        self.scale = scale
        if isinstance(ranges, list) and isinstance(ranges[0], list):
            pass
        else:
            ranges = [ranges]
        xy_angle_ = []
        for _ranges in ranges:
            xy_angle = np.linspace(_ranges[0], _ranges[1], _ranges[2])
            xy_angle_.append(xy_angle)
        xy_angle = np.concatenate(xy_angle_, axis=0)
        xy_angle = np.deg2rad(xy_angle)
        if isinstance(radius, list) and isinstance(radius[0], list):
            pass
        else:
            radius = [radius]
        radius_ = []
        for _radius in radius:
            radius = np.logspace(np.log10(_radius[0]), np.log10(_radius[1]), _radius[2])
            radius_.append(radius)
        radius = np.concatenate(radius_, axis=0)
        if isinstance(lookat, list) and isinstance(lookat[0], list):
            start = np.array(lookat[0])
            end = np.array(lookat[1])
            t = np.linspace(0, 1, radius.shape[0])
            lookat = start[None] * (1 - t[:, None]) + end[None] * t[:, None]
        else:
            lookat = np.array(lookat)[None].repeat(len(radius), 0)
        if isinstance(angle, list):
            Rrel = []
            angle = np.linspace(angle[0], angle[1], radius.shape[0])
            for _angle in angle:
                Rrel.append(cv2.Rodrigues(np.deg2rad(np.array([_angle, 0., 0.])))[0])
            Rrel = np.stack(Rrel)
        else:
            Rrel = cv2.Rodrigues(np.deg2rad(np.array([angle, 0., 0.])))[0]
            Rrel = Rrel[None].repeat(len(radius), 0)
        height = radius * np.cos(np.deg2rad(angle))
        radius2d = radius * np.sin(np.deg2rad(angle))
        x_ = radius2d * np.sin(xy_angle) + lookat[:, 0]
        y_ = radius2d * np.cos(xy_angle) + lookat[:, 1]
        z_ = np.zeros_like(x_) + lookat[:, 2] - height

        center = np.stack([x_, y_, z_], axis=-1).reshape(-1, 3, 1).astype(np.float32)
        # make the zaxis lookat the center
        zaxis = lookat - center.reshape(-1, 3)
        zaxis = zaxis / np.linalg.norm(zaxis, axis=-1, keepdims=True)
        # xaxis: parallel to the ground
        world_up = np.array([[0., 0., -1.]])
        right = np.cross(zaxis, world_up)
        right = right / np.linalg.norm(right, axis=-1, keepdims=True)
        down = np.cross(zaxis, right)
        down = down / np.linalg.norm(down, axis=-1, keepdims=True)
        if False:
            yaxis = np.stack([np.sin(xy_angle), np.cos(xy_angle), np.zeros_like(xy_angle)], axis=-1)
            xaxis = np.cross(yaxis, zaxis)
            xaxis = xaxis / np.linalg.norm(xaxis, axis=-1, keepdims=True)
            R = np.stack([xaxis, yaxis, zaxis], axis=-1)
        else:
            R = np.zeros((xy_angle.shape[0], 3, 3))
            R[:, 0, :] = right
            R[:, 1, :] = down
            R[:, 2, :] = zaxis
        infos = []
        for i in range(center.shape[0]):
            _R = R[i]
            T = - _R @ center[i]
            camera={
                'K': self.K, 'R': _R, 'T': T,
                'H': self.H, 'W': self.W,
                'center': center[i]
            }
            infos.append({
                'camera': camera,
                'scale': scale
            })
        self.infos = infos