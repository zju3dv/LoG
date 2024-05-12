import os
from LoG.render.renderer import NaiveRendererAndLoss
import numpy as np
import cv2
import torch
from tqdm import tqdm
from LoG.utils.config import load_object, Config
from LoG.utils.command import update_global_variable
from LoG.utils.trainer import prepare_batch

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
    device = torch.device('cuda')
    # Load naive model and point cloud
    from LoG.model.base_gaussian import BaseGaussian
    plyname = cfg['PLYNAME']
    from LoG.utils.file import read_ply_and_log
    xyz, colors = read_ply_and_log(plyname, cfg['scale3d'])
    scaling = cfg.radius + np.zeros((xyz.shape[0], 3), dtype=np.float32)
    opacity = 0.9 + np.zeros((xyz.shape[0], 1), dtype=np.float32)
    rotation = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    rotation[:, 0] = 1.
    record = {
        'xyz': xyz,
        'colors': colors,
        'scaling': scaling,
        'opacity': opacity,
        'rotation': rotation,
    }
    model = BaseGaussian.create_from_record(record)
    renderer = NaiveRendererAndLoss(use_origin_render=False, background = [1., 1., 1.])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.to(device)
    renderer.to(device)
    for i, batch in enumerate(tqdm(dataloader)):
        batch = prepare_batch(batch, device)
        with torch.no_grad():
            output = renderer.vis(batch, model)
        render = output['render'][0]
        vis = renderer.tensor_to_bgr(render)
        outname = f'debug/{i:06d}.jpg'
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, vis)
    renderer.make_video('debug')
