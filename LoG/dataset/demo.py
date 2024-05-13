import os
import numpy as np
import cv2
from .camera_utils import read_cameras
from .base import prepare_camera
from scipy.spatial.transform import Rotation

class DemoBase:
    def __init__(self, znear=0.01, zfar=100.):
        self.znear = znear
        self.zfar = zfar

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        data = self.infos[index]
        camera = prepare_camera(data['camera'], data['scale'], self.znear, self.zfar)
        ret = {
            'index': index,
            'camera': camera
        }
        return ret

def create_center_radius(center, radius=5., up='y', ranges=[0, 360, 36], angle_x=0, **kwargs):
    center = np.array(center).reshape(1, 3)
    thetas = np.deg2rad(np.linspace(*ranges))
    st = np.sin(thetas)
    ct = np.cos(thetas)
    zero = np.zeros_like(st)
    Rotx = cv2.Rodrigues(np.deg2rad(angle_x) * np.array([1., 0., 0.]))[0]
    if up == 'z':
        center = np.stack([radius*ct, radius*st, zero], axis=1) + center
        R = np.stack([-st, ct, zero, zero, zero, zero-1, -ct, -st, zero], axis=-1)
    elif up == 'y':
        center = np.stack([radius*ct, zero, radius*st, ], axis=1) + center
        R = np.stack([
            +st,  zero,  -ct,
            zero, zero-1, zero, 
            -ct,  zero, -st], axis=-1)
    R = R.reshape(-1, 3, 3)
    R = np.einsum('ab,fbc->fac', Rotx, R)
    center = center.reshape(-1, 3, 1)
    T = - R @ center
    RT = np.dstack([R, T])
    return RT

class DemoDataset(DemoBase):
    def __init__(self, size = 2048, znear=0.1, zfar=100., 
                 radius=3., ranges=[0, 360, 45], 
                 center=[0, 0, 0.],
                 focal=-1,
                 focal_scale=1.) -> None:
        super().__init__()
        if focal == -1:
            focal = size * focal_scale
        K = np.array([
            [focal, 0, size//2], 
            [0, focal, size//2], 
            [0, 0, 1]])
        RT = create_center_radius(
            center, radius=radius, up='z', ranges=ranges, angle_x=0)
        infos = []
        for i in range(RT.shape[0]):
            infos.append({
                'camera': {
                'R': RT[i, :3, :3],
                'T': RT[i, :3, 3:4],
                'K': K,
                'W': size,
                'H': size,
                'center': - RT[i, :3, :3].T @ RT[i, :3, 3:4],
                },
                'scale': 1,
            })
        # add zoom in
        self.infos = infos
        self.znear = znear
        self.zfar = zfar

class GivenTrajs(DemoBase):
    def __init__(self, cameras, znear=0.01, zfar=100, scale3d=1.) -> None:
        super().__init__(znear, zfar)
        from .camera_utils import read_cameras
        cameras = read_cameras(cameras)
        camera_vis_all = []
        for key, camera in cameras.items():
            camera['T'] *= scale3d
            center = - camera['R'].T @ camera['T']
            info = {
                'camera': {
                    'K': camera['K'], 'R': camera['R'], 'T': camera['T'],
                    'H': camera['H'], 'W': camera['W'],
                    'center': center
                },
                'scale': 4,
            }
            camera_vis_all.append(info)
        self.infos = camera_vis_all

class ComposeDataset(DemoBase):
    def __init__(self, datasets):
        super().__init__()
        from LoG.utils.config import load_object
        datasets_all = []
        length = 0
        for dataset in datasets:
            _dataset = load_object(dataset['module'], dataset['args'])
            datasets_all.extend(_dataset.infos)
            length += len(_dataset)
        self.infos = datasets_all

class ZoomInOut(DemoBase):
    def __init__(self, cameras, sub, zranges, 
                 scale=1, steps=100,
                 znear=0.01, zfar=100.,
                 direction=[0., 0., 1.],
                 H=-1, W=-1,
                 use_logspace=True,
                 ) -> None:
        cameras = read_cameras(cameras)
        camera = cameras[sub]
        zdir = np.array(direction).reshape(3, 1)
        zdir = zdir / np.linalg.norm(zdir)
        zdir = camera['R'].T @ zdir
        if use_logspace:
            zranges = np.log(np.linspace(np.exp(zranges[0]), np.exp(zranges[1]), steps))
        else:
            zranges = np.linspace(zranges[0], zranges[1], steps)
        infos = []
        if H == -1:
            H = camera['H']
        if W == -1:
            W = camera['W']
        for i in range(zranges.shape[0]):
            R = camera['R']
            T = camera['T']
            center_old = -R.T @ T
            center_new = center_old + zdir * zranges[i]
            T = -R @ center_new
            camera_new = {
                'R': R,
                'T': T,
                'K': camera['K'],
                'H': H,
                'W': W,
                'center': center_new
            }
            infos.append({'camera': camera_new, 'scale': scale})
        self.infos = infos
        self.znear = znear
        self.zfar = zfar

class ShowLevel(DemoBase):
    def __init__(self, cameras, sub, steps=300, scale=1, znear=0.01, zfar=100, mode='level'):
        super().__init__(znear, zfar)
        cameras = read_cameras(cameras)
        camera = cameras[sub]
        self.pixel_max = 6
        self.mode = mode
        infos = []
        for i in range(steps):
            R = camera['R']
            T = camera['T']
            center = -R.T @ T
            camera_new = {
                'R': R,
                'T': T,
                'K': camera['K'],
                'H': camera['H'],
                'W': camera['W'],
                'center': center
            }
            infos.append({'camera': camera_new, 'scale': scale})
        self.infos = infos
    
    def __getitem__(self, index):
        data = self.infos[index]
        camera = prepare_camera(data['camera'], data['scale'], self.znear, self.zfar)
        ret = {
            'index': index,
            'camera': camera,
        }
        if self.mode == 'pixel':
            ret['model_state'] = {'min_resolution_pixel': 2**((1 - index/len(self))*self.pixel_max)}
        else:
            ret['model_state'] = {'current_depth': index}
        return ret

class GivenKRCenter(DemoBase):
    def __init__(self, K, R, center, H, W, steps, scale=1):
        self.znear = 0.01
        self.zfar = 100.
        K = np.array(K)
        R = np.array(R)
        center = np.array(center)
        # padding K, R, T
        timestep = np.linspace(0, 1, steps)
        if len(K.shape) == 2:
            K = K[None].repeat(steps, axis=0)
        elif len(K.shape) == 3 and K.shape[0] == 2:
            # interpolate
            K = np.stack([K[0] + (K[1] - K[0]) * t for t in timestep])
        else:
            assert K.shape[0] == steps, f'K shape {K.shape} not match steps {steps}'
        if len(R.shape) == 2:
            R = R[None].repeat(steps, axis=0)
        elif len(R.shape) == 3 and R.shape[0] == 2:
            # interpolate
            R = np.stack([R[0] + (R[1] - R[0]) * t for t in timestep])
        else:
            assert R.shape[0] == steps, f'R shape {R.shape} not match steps {steps}'
        if len(center.shape) == 1:
            center = center[None].repeat(steps, axis=0)
        elif len(center.shape) == 2 and center.shape[0] == 2:
            # interpolate
            center = np.stack([center[0] + (center[1] - center[0]) * t for t in timestep])
        else:
            assert center.shape[0] == steps, f'center shape {center.shape} not match steps {steps}'
        infos = []
        import ipdb; ipdb.set_trace()
        for i in range(steps):
            camera = {
                'K': K[i],
                'R': R[i],
                'T': -R[i] @ center[i].reshape(3, 1),
                'H': H,
                'W': W,
                'center': center[i].reshape(3, 1)
            }
            infos.append({'camera': camera, 'scale': scale})
            print(infos[-1])
        self.infos = infos

class InterpolatingExtrinsics:
    def __init__(self, c2w: np.ndarray) -> None:
        self.Q = Rotation.from_matrix(c2w[..., :3, :3]).as_quat()
        self.T = c2w[..., :3, 3]

    def __add__(lhs, rhs):  # FIXME: Dangerous
        Ql, Qr = lhs.Q, rhs.Q
        Qr = np.where((Ql * Qr).sum(axis=-1, keepdims=True) < 0, -Qr, Qr)
        lhs.Q = Ql + Qr
        lhs.T = lhs.T + rhs.T
        return lhs

    def __radd__(rhs, lhs):
        return rhs.__add__(lhs)

    def __mul__(lhs, rhs: np.ndarray):
        lhs.Q = rhs[..., None] * lhs.Q
        lhs.T = rhs[..., None] * lhs.T
        return lhs  # inplace modification

    def __rmul__(rhs, lhs: np.ndarray):
        return rhs.__mul__(lhs)

    def numpy(self):
        return np.concatenate([Rotation.from_quat(self.Q).as_matrix(), self.T[..., None]], axis=-1).astype(np.float32)

def cubic_spline(us: np.ndarray, N: int):
    if isinstance(us, int) or isinstance(us, float): us = np.asarray([us])
    if isinstance(us, list): us = np.asarray(us)

    # Preparation
    t = (N - 1) * us  # expanded to the length of the sequence
    i0 = np.floor(t).astype(np.int32) - 1
    i0 = np.where(us != 1.0, i0, i0 - 1)  # remove end point (nans for 1s)
    i1 = i0 + 1
    i2 = i0 + 2
    i3 = i0 + 3
    i0, i1, i2, i3 = np.clip(i0, 0, N - 1), np.clip(i1, 0, N - 1), np.clip(i2, 0, N - 1), np.clip(i3, 0, N - 1)
    t0, t1, t2, t3 = i0 / (N - 1), i1 / (N - 1), i2 / (N - 1), i3 / (N - 1)
    t = (t - i1)  # normalize to the start?
    t = t.astype(np.float32)  # avoid fp64 problems

    # Compute coeffs
    tt = t * t
    ttt = tt * t
    a = (1 - t) * (1 - t) * (1 - t) * (1. / 6.)
    b = (3. * ttt - 6. * tt + 4.) * (1. / 6.)
    c = (-3. * ttt + 3. * tt + 3. * t + 1.) * (1. / 6.)
    d = ttt * (1. / 6.)

    t0, t1, t2, t3 = t0.astype(np.float32), t1.astype(np.float32), t2.astype(np.float32), t3.astype(np.float32)
    a, b, c, d = a.astype(np.float32), b.astype(np.float32), c.astype(np.float32), d.astype(np.float32)

    return t, (i0, i1, i2, i3), (t0, t1, t2, t3), (a, b, c, d)

def gen_cubic_spline_interp_func(c2ws: np.ndarray, smoothing_term=10.0, *args, **kwargs):
    # Split interpolation
    N = len(c2ws)
    assert N > 3, 'Cubic Spline interpolation requires at least four inputs'
    if smoothing_term == 0:
        low = -2  # when we view index as from 0 to n, should remove first two segments
        high = N - 1 + 4 - 2  # should remove last one segment, please just work...
        c2ws = np.concatenate([c2ws[-2:], c2ws, c2ws[:2]])

    def lf(us: np.ndarray):
        N = len(c2ws)  # should this be recomputed?
        t, (i0, i1, i2, i3), (t0, t1, t2, t3), (a, b, c, d) = cubic_spline(us, N)

        # Extra inter target
        c0, c1, c2, c3 = InterpolatingExtrinsics(c2ws[i0]), InterpolatingExtrinsics(c2ws[i1]), InterpolatingExtrinsics(c2ws[i2]), InterpolatingExtrinsics(c2ws[i3])
        c = c0 * a + c1 * b + c2 * c + c3 * d  # to utilize operator overloading
        c = c.numpy()  # from InterpExt to numpy
        if isinstance(us, int) or isinstance(us, float): c = c[0]  # remove extra dim
        return c

    if smoothing_term == 0:
        def pf(us): return lf((us * N - low) / (high - low))  # periodic function will call the linear function
        f = pf  # periodic function
    else:
        f = lf  # linear function
    return f

def interpolate_camera_path(c2ws: np.ndarray, steps=50, smoothing_term=10.0, **kwargs):
    # Store interpolation parameters
    f = gen_cubic_spline_interp_func(c2ws, smoothing_term)

    # The interpolation t
    us = np.linspace(0, 1, steps, dtype=c2ws.dtype)
    return f(us)

class InterpolatePath(DemoBase):
    def __init__(self, cameras, subs=[], steps=300, znear=0.1, zfar=100., 
        scale=1, scale3d=1.,
        H=-1, W=-1, ref_cam=None,
        ) -> None:
        super().__init__(znear=znear, zfar=zfar)
        if os.path.isdir(cameras):
            cameras = read_cameras(cameras)
        elif os.path.isfile(cameras):
            cameras = read_cameras(os.path.dirname(cameras))
        Rlist = []
        Tlist = []
        if len(subs) == 0:
            subs = list(cameras.keys())
        for sub in subs:
            if isinstance(sub, str):
                Rlist.append(cameras[sub]['R'])
                Tlist.append(cameras[sub]['T'][:, 0])
            elif isinstance(sub, dict):
                R = cameras[sub['name']]['R']
                T = cameras[sub['name']]['T'][:, 0]
                center = - R.T @ T[:, None]
                if 'rotate_axis' in sub:
                    if sub['rotate_axis'] == 'z':
                        rotation = cv2.Rodrigues(np.deg2rad(np.array([0., 0., sub['rotate_angle']])))[0]
                    if sub['rotate_axis'] == 'x':
                        rotation = cv2.Rodrigues(np.deg2rad(np.array([sub['rotate_angle'], 0., 0.])))[0]               
                    print(rotation)
                    R = rotation @ R
                    T = (-R @ center)[:, 0]
                if 'translation' in sub:
                    translation = np.array(sub['translation']).reshape(3, 1) / scale3d
                    center = center + translation
                    print(center.T)
                    T = (-R @ center)[:, 0]
                Rlist.append(R)
                Tlist.append(T)
        Rlist = np.stack(Rlist)
        Tlist = np.stack(Tlist) * scale3d
        centerlist = np.einsum('ijk,ik->ij', Rlist.transpose(0, 2, 1), -Tlist)
        if False:
            Rresample, Tresample = interpolate_camera_trajectory(Rlist, centerlist, num_samples=steps, resample=False)
        elif True:
            Rlist = np.einsum('ijk->ikj', Rlist)
            RTlist = np.dstack([Rlist, centerlist])
            RTlist = interpolate_camera_path(RTlist, steps=steps, smoothing_term=5.)
            Rresample = RTlist[:, :3, :3].transpose(0, 2, 1)
            Tresample = RTlist[:, :3, 3:] 
        else:
            Rresample, Tresample = Rlist, cenerlist
        infos = []
        if ref_cam is None:
            ref_cam = list(cameras.keys())[0]
        K = cameras[ref_cam]['K']
        if H == -1:
            H = cameras[list(cameras.keys())[0]]['H']
        if W == -1:
            W = cameras[list(cameras.keys())[0]]['W']
        for i in range(Rresample.shape[0]):
            R = Rresample[i]
            center = Tresample[i].reshape(3, 1)
            T = -R @ center
            camera_new = {
                'R': R,
                'T': T,
                'K': K,
                'H': H,
                'W': W,
                'center': center
            }
            infos.append({'camera': camera_new, 'scale': scale})
        self.infos = infos