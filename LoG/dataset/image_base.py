import os
from os.path import join
import numpy as np
import cv2

class ImageBase:
    def __init__(self, cache=None, cameras='', namelist=None, ignorelist=None, znear=0.01, zfar=100., offset=[0., 0., 0.]) -> None:
        self.cache = cache
        self.cameras = cameras
        if namelist is not None:
            if isinstance(namelist, str):
                if os.path.exists(namelist):
                    with open(self.namelist, 'r') as f:
                        namelist = f.readlines()
        self.namelist = namelist            
        self.ignorelist = ignorelist
        self.offset = np.array(offset, dtype=np.float32).reshape(3, 1)
        self.use_cache = False
        self.read_img = True
        self.znear = znear
        self.zfar = zfar
        self.partial_indices = None
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def write_cache(self, infos, name='cache'):
        if not name.endswith('.pkl'):
            cachename = join(self.cache, name + '.pkl')
        else:
            cachename = name
        if not os.path.exists(cachename):
            print('write cache to ', cachename)
            import pickle
            os.makedirs(os.path.dirname(cachename), exist_ok=True)
            with open(cachename, 'wb') as f:
                pickle.dump(infos, f)
            # joblib.dump(infos, cachename)
    
    def read_cache(self, name='cache'):
        if not name.endswith('.pkl'):
            cachename = join(self.cache, name + '.pkl')
        else:
            cachename = name
        if os.path.exists(cachename):
            import pickle
            with open(cachename, 'rb') as f:
                return True, pickle.load(f)
        return False, None

    def set_partial_indices(self, partial):
        self.partial_indices = partial
        print(f'[{self.__class__.__name__}] set partial indices {len(partial)}')

    @staticmethod
    def make_video(path, remove_image=False, fps=30):
        cmd = f'/usr/bin/ffmpeg -y -r {fps} -i {path}/%06d.jpg  -vf scale="2*ceil(iw/2):2*ceil(ih/2)" -vcodec libx264 -r {fps} {path}.mp4 -loglevel quiet'
        print(cmd)
        os.system(cmd)

    def check_cameras(self, scale3d=-1, scale_camera_K=1.):
        from .camera_utils import read_cameras
        cameras = read_cameras(join(self.root, self.cameras))
        print('Loaded {} cameras from {}'.format(len(cameras), join(self.root, self.cameras)))
        if self.namelist is not None:
            cameras_new = {}
            for name in self.namelist:
                name = name.strip()
                cameras_new[name] = cameras[name]
            cameras = cameras_new
        if self.ignorelist is not None:
            with open(self.ignorelist, 'r') as f:
                ignorelist = f.readlines()
            for name in ignorelist:
                name = name.strip()
                cameras.pop(name)
        print('scale3d = {}'.format(scale3d))
        if scale3d > 0:
            for camname, camera in cameras.items():
                center = - np.dot(camera['R'].T, camera['T'] * scale3d) - self.offset
                T = - camera['R'] @ center
                camera['center'] = center
                camera['T'] = T
        if scale_camera_K != 1.:
            for camname, camera in cameras.items():
                camera['K'][:2, :] *= scale_camera_K
                camera['W'] = int(scale_camera_K * camera['W'])
                camera['H'] = int(scale_camera_K * camera['H'])
        return cameras

    @staticmethod
    def read_image(imgname):
        assert os.path.exists(imgname), imgname
        img = cv2.imread(imgname)
        assert img is not None, imgname
        img = img.astype(np.float32)/255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def read_image_with_cache(self, imgname):
        if self.use_cache:
            if imgname in self.cache.keys():
                return self.cache[imgname]
            img = self.read_image(imgname)
            self.cache[imgname] = img
            return img
        else:
            return self.read_image(imgname)
    
    def read_depth(self, depthname):
        assert os.path.exists(depthname), depthname
        depth = cv2.imread(depthname, -1)
        assert depth is not None, depthname
        depth = depth.astype(np.float32) / (2**16 - 1)
        return depth
    
    def read_mask(self, mskname):
        assert os.path.exists(mskname), mskname
        msk = cv2.imread(mskname, -1)
        assert msk is not None, mskname
        msk = msk.astype(np.float32) / 255.0
        return msk
    
    @staticmethod
    def crop_image(img, crop_size):
        xranges = np.arange(0, img.shape[1] - crop_size[1] + 1)
        yranges = np.arange(0, img.shape[0] - crop_size[0] + 1)
        sample_x = int(np.random.choice(xranges))
        sample_y = int(np.random.choice(yranges))
        l, t, r, b = sample_x, sample_y, sample_x + crop_size[1], sample_y + crop_size[0]
        return l, t, r, b