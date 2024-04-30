import numpy as np
import cv2
import torch
from LoG.utils.config import load_object, Config
from LoG.utils.command import update_global_variable

if __name__ == '__main__':
    usage = 'test dataset'
    args, cfg = Config.load_args(usage=usage)
    cfg = update_global_variable(cfg, cfg)

    if 'dataset' in cfg.split:
        dataset = cfg[cfg.split]
    else:
        dataset = cfg[cfg.split].dataset
    dataset = load_object(dataset.module, dataset.args)
    print('dataset: ', len(dataset))

    for i in range(len(dataset)):
        data = dataset[i]
        if 'image' in data.keys():
            vis = data['image']
        print('data keys: ', data.keys())
        print('imgname: ', data['imgname'])
        print('image shape: ', data['image'].shape)
        vis = (vis[:, :, ::-1] * 255).astype(np.uint8)
        outname = f'debug/{i:06d}.jpg'
        cv2.imwrite(outname, vis)