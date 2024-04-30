import numpy as np
from ..utils.camera import focal2fov, getProjectionMatrix2

def rescale_camera(camera, scale, H=-1, W=-1):
    camera = camera.copy()
    if W == -1:
        W = int(camera['W'] / scale)
    if H == -1:
        H = int(camera['H'] / scale)
    K = camera['K'].copy()
    scale_x = W / camera['W']
    scale_y = H / camera['H']
    K[0, :] *= scale_x
    K[1, :] *= scale_y
    camera['W'] = W
    camera['H'] = H
    camera['K'] = K
    return camera

def prepare_camera(camera, scale, znear, zfar):
    ret = {}
    # w, h
    ret['image_width'] = int(camera['W'] / scale)
    ret['image_height'] = int(camera['H'] / scale)
    # calculate the accurate scale_x, y
    scale_x = ret['image_width'] / camera['W']
    scale_y = ret['image_height'] / camera['H']
    focal_length_y = camera['K'][1, 1] * scale_y
    focal_length_x = camera['K'][0, 0] * scale_x
    ret['FoVy'] = focal2fov(focal_length_y, camera['H'] * scale_y)
    ret['FoVx'] = focal2fov(focal_length_x, camera['W'] * scale_x)
    ret['K'] = camera['K'].copy()
    ret['K'][0, :] *= scale_x
    ret['K'][1, :] *= scale_y
    # calculate the project matrix considering the cx, cy
    ret['projection_matrix'] = getProjectionMatrix2(
        K=ret['K'], H=ret['image_height'], W=ret['image_width'],
        znear=znear, zfar=zfar).T
    
    world_view_transform = np.eye(4)
    world_view_transform[:3, :3] = camera['R']
    world_view_transform[:3, 3:] = camera['T']
    world_view_transform = world_view_transform.T
    ret['camera_center'] = camera['center'].reshape(3,)
    ret['world_view_transform'] = world_view_transform
    ret['full_proj_transform'] = world_view_transform @ ret['projection_matrix']
    ret['znear'] = znear
    ret['zfar'] = zfar
    ret['R'] = camera['R']
    ret['T'] = camera['T']
    ret['scale'] = scale
    for key, val in ret.items():
        if isinstance(val, np.ndarray):
            ret[key] = val.astype(np.float32)
    return ret