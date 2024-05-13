import os
from os.path import join
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from .image_base import ImageBase
from .base import prepare_camera, rescale_camera
from .camera_utils import get_center_and_diag

def read_undistort_rescale_write(info):
    flag_read_img = False
    for scale in info['scales']:
        cachename = join(info['cache'], str(scale), info['imgname'])
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        if not os.path.exists(cachename):
            flag_read_img = True
            break
    else:
        return 0
    imgname = join(info['root'], info['imgname'])
    assert os.path.exists(imgname), imgname
    camera = info['camera']
    if flag_read_img:
        # read the image with pillow, because opencv ignore the orientation of image
        # img = cv2.imread(imgname)
        img = Image.open(imgname)
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        assert img.shape[0] == camera['H'] and img.shape[1] == camera['W'], f'{imgname}: {img.shape} != {camera["H"]}, {camera["W"]}'
        if 'mapx' in info['camera'].keys() and 'mapy' in info['camera'].keys():
            mapx, mapy = info['camera']['mapx'], info['camera']['mapy']
        else:
            width, height = camera['W'], camera['H']
            newK, roi = cv2.getOptimalNewCameraMatrix(camera['K'], camera['dist'], 
                        (width, height), 0, (width,height), centerPrincipalPoint=True)
            mapx, mapy = cv2.initUndistortRectifyMap(camera['K'], camera['dist'], None, newK, (width, height), 5)
            camera['K'] = newK
        if mapx is not None and mapy is not None:
            img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    else:
        if 'mapx' not in info['camera'].keys():
            width, height = camera['W'], camera['H']
            newK, roi = cv2.getOptimalNewCameraMatrix(camera['K'], camera['dist'], 
                        (width, height), 0, (width,height), centerPrincipalPoint=True)
            mapx, mapy = cv2.initUndistortRectifyMap(camera['K'], camera['dist'], None, newK, (width, height), 5)
            camera['K'] = newK

    for scale in info['scales']:
        cachename = join(info['cache'], str(scale), info['imgname'])
        if os.path.exists(cachename):
            continue
        camera_scale = camera.copy()
        camera_scale['K'] = camera['K'].copy()
        W = int(camera['W'] / scale)
        H = int(camera['H'] / scale)
        dst = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        cv2.imwrite(cachename, dst)
    return 0

class ImageDataset(ImageBase):
    @staticmethod
    def init_camera(camera):
        width, height = camera['W'], camera['H']
        assert width != 0 and height != 0, f'width or height is 0: {width}, {height}'
        dist = camera['dist']
        if np.linalg.norm(dist) < 1e-5:
            mapx, mapy = None, None
            newK = camera['K'].copy()
        else:
            newK, roi = cv2.getOptimalNewCameraMatrix(camera['K'], camera['dist'], 
                        (width, height), 0, (width,height), centerPrincipalPoint=True)
            mapx, mapy = cv2.initUndistortRectifyMap(camera['K'], camera['dist'], None, newK, (width, height), 5)
        return mapx, mapy, newK

    def check_undis_camera(self, camname, cameras_cache, camera_undis, share_camera=False):
        if share_camera:
            cache_camname = 'cache'
        else:
            if '/' in camname:
                cache_camname = camname.split('/')[0]
            else:
                cache_camname = camname

        if cache_camname not in cameras_cache:
            print(f'[{self.__class__.__name__}] init camera {cache_camname}')
            cameras_cache[cache_camname] = self.init_camera(camera_undis)
        mapx, mapy, newK = cameras_cache[cache_camname]
        camera = {
            'K': newK,
            'mapx': mapx,
            'mapy': mapy
        }
        for key in ['R', 'T', 'W', 'H', 'center']:
            camera[key] = camera_undis[key]
        return camera

    def __init__(self, root, cameras='sparse/0', scales=[1,2,4], 
                scale3d=1., ext='.JPG', images='images', scale_camera_K=1., 
                mask_ignore=None,
                 pre_undis=True, share_camera=False, crop_size=[-1, -1],
                 crop_ltrb=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.root = os.path.abspath(root)
        self.cameras = cameras
        self.image_dir = images
        self.ext = ext
        self.mask_ignore = mask_ignore
        self.scales = scales
        self.downsample_scale = 1
        self.scale3d = scale3d
        self.crop_size = crop_size
        self.crop_ltrb = crop_ltrb
        print(f'[{self.__class__.__name__}] set scales: {scales}, crop size: {crop_size}')
        if self.cache is None:
            self.cache = join(self.root, 'cache')
            cachedir = self.cache
        else:
            if self.cache.endswith('.pkl'):
                cachedir = join(self.root, self.cache.replace('.pkl', ''))
            else:
                cachedir = join(self.root, self.cache)
        self.cachedir = cachedir
        print(f'[{self.__class__.__name__}] cache dir: {self.cachedir}')
        flag, infos = self.read_cache(name=cachedir+'.pkl')
        if not flag:
            cameras = self.check_cameras(scale3d=scale3d, scale_camera_K=scale_camera_K)
            # undistort and scale
            cameras_cache = {}
            infos = []
            for camname, camera_dis in cameras.items():
                if pre_undis:
                    camera = self.check_undis_camera(camname, cameras_cache, camera_dis, share_camera)
                else:
                    camera = camera_dis
                camera_ = camera.copy()
                # camera_.pop('mapx')
                # camera_.pop('mapy')
                imgname = join(self.root, images, camname + ext)
                if not os.path.exists(imgname):
                    print('Not exists:', imgname)
                    continue
                infos.append({
                    'root': self.root,
                    'cache': cachedir,
                    'imgname': join(images, camname + ext),
                    'camera': camera_,
                    'scales': scales
                })
            print(f'[{self.__class__.__name__}] undistort and scale {len(infos)} images ')
            for info in tqdm(infos):
                read_undistort_rescale_write(info)
                info['camera'].pop('mapx', None)
                info['camera'].pop('mapy', None)
            self.write_cache(infos, name=cachedir+'.pkl')
        
        centers = np.stack([-info['camera']['R'].T @ info['camera']['T'] for info in infos], axis=0)
        offset, radius = get_center_and_diag(centers)
        print(f'[{self.__class__.__name__}] offset: {offset}, radius: {radius}')
        self.current_scale = scales[-1]
        self.infos = infos
        print(f'[{self.__class__.__name__}] init dataset with {len(infos)} images')

    def set_state(self, scale=None, crop_size=None, downsample_scale=1, namelist=None):
        if scale is not None:
            assert scale in self.scales, f'scale {scale} not in {self.scales}'
            self.current_scale = scale
        self.downsample_scale = downsample_scale
        if crop_size is not None:
            print(f'[{self.__class__.__name__}] set crop size {crop_size}')
            self.crop_size = crop_size
        print(f'[{self.__class__.__name__}] set scale {scale}, crop_size: {self.crop_size}, downsample_scale: {downsample_scale}')

    def __len__(self):
        if self.partial_indices is None:
            return len(self.infos)
        else:
            return len(self.partial_indices)

    def crop_image(self, img, crop_size):
        if isinstance(img, str):
            pass
        xranges = np.arange(0, img.shape[1] - crop_size[1] + 1)
        yranges = np.arange(0, img.shape[0] - crop_size[0] + 1)
        sample_x = int(np.random.choice(xranges))
        sample_y = int(np.random.choice(yranges))
        l, t, r, b = sample_x, sample_y, sample_x + crop_size[1], sample_y + crop_size[0]
        return l, t, r, b

    def update_crop(self, img, camera, l, t, r, b):
        camera['K'] = camera['K'].copy()
        img = img[t:b, l:r]
        camera['K'][0, 2] -= l
        camera['K'][1, 2] -= t
        camera['W'] = r - l
        camera['H'] = b - t
        return img, camera

    def __getitem__(self, index):
        if self.partial_indices is None:
            true_index = index
        else:
            true_index = self.partial_indices[index]
        data = self.infos[true_index]
        imgname = data['imgname']
        imgname = join(self.cachedir, str(self.current_scale), imgname)
        if self.read_img and os.path.exists(imgname):
            img = self.read_image_with_cache(imgname)
        else:
            img = imgname
        if self.downsample_scale != 1:
            scale = self.downsample_scale * self.current_scale
            camera = rescale_camera(data['camera'], scale)
            if self.read_image:
                # cv2.INTER_AREA for anti-alias resize
                img = cv2.resize(img, (camera['W'], camera['H']), interpolation=cv2.INTER_AREA)
        else:
            camera = rescale_camera(data['camera'], self.current_scale)
        # check mask
        msk = None
        if self.mask_ignore is not None:
            mskname = join(self.root, self.mask_ignore['path'], data['imgname'].replace(self.ext, '.png'))
            if self.read_img and os.path.exists(mskname):
                msk = self.read_mask(mskname)
                if self.current_scale != self.mask_ignore.scale:
                    import ipdb;ipdb.set_trace()
                if self.mask_ignore['type'] == 'background':
                    # dilate
                    border = int(msk.shape[0]//50) * 2 + 1
                    kernel = np.ones((border, border), np.float32)
                    msk = cv2.dilate(msk, kernel)
                    msk = 1 -  msk
        
        if self.crop_ltrb is not None and not isinstance(img, str):
            l, t, r, b = self.crop_ltrb
            img, camera = self.update_crop(img, camera, l, t, r, b)
        elif self.crop_size[0] > 0 and self.crop_size[1] > 0 and not isinstance(img, str):
            l, t, r, b = self.crop_image(img, self.crop_size)
            img, camera = self.update_crop(img, camera, l, t, r, b)
        camera = prepare_camera(camera, scale=1, znear=self.znear, zfar=self.zfar)
        ret = {
            'image': img,
            'imgname': imgname,
            'index': index,
            'true_index': true_index,
            'camera': camera,
        }
        if msk is not None:
            ret['mask_ignore'] = msk
        ret.update(data.get('extra', {}))
        return ret

class DepthDataset(ImageDataset):
    def __init__(self, depth_scale, depth_dir='depth', **kwargs):
        super().__init__(**kwargs)
        self.depth_scale = depth_scale
        self.depth_dir = depth_dir
    
    def __getitem__(self, index):
        ret = super().__getitem__(index)
        depthname = ret['imgname'].replace(
            self.image_dir, self.depth_dir)\
            .replace(f'{os.sep}{self.current_scale}{os.sep}{self.depth_dir}', f'{os.sep}{self.depth_scale}{os.sep}{self.depth_dir}')\
            + '.png'
        if self.read_image:
            # depth: (0.0 -> 1.0)
            depth = self.read_depth(depthname)
        else:
            depth = depthname
        ret['depth'] = depth
        return ret