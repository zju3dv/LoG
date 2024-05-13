import os
from os.path import join
import numpy as np
from tqdm import tqdm
from LoG.utils.config import load_object, Config
from LoG.utils.command import update_global_variable, load_statedict, copy_git_tracked_files
import cv2
import torch

def demo(cfg, model, device):
    dataset = load_object(cfg[cfg.split].dataset.module, cfg[cfg.split].dataset.args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # prepare the renderer
    if 'render' in cfg[cfg.split]:
        renderer = load_object(cfg[cfg.split].render.module, cfg[cfg.split].render.args)
    else:
        renderer = load_object(cfg.train.render.module, cfg.train.render.args)
        renderer.split = 'demo'
    renderer.to(device)
    model.to(device)
    model.eval()
    if 'model_state' in cfg[cfg.split]:
        model.set_state(**cfg[cfg.split]['model_state'])
    if 'render_state' in cfg[cfg.split]:
        renderer.set_state(**cfg[cfg.split]['render_state'])
    from LoG.utils.trainer import prepare_batch
    from tqdm import tqdm
    total_time = 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    render_type = cfg.get('render_type', 'rgb')
    if render_type == 'depth':
        renderer.render_depth = True
        depth_min = cfg.get('depth_min', 0.01)
        depth_max = cfg.get('depth_max', 10.)
    elif render_type == 'height':
        renderer.render_depth = True
        height_min = cfg.get('height_min', 0.01)
        height_max = cfg.get('height_max', 10.)

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = prepare_batch(batch, device)
        with torch.no_grad():
            output = renderer.vis(batch, model)
        if batch_idx > 10:
            break

    for batch in tqdm(dataloader):
        batch = prepare_batch(batch, device)
        if 'model_state' in batch:
            model.set_state(**batch['model_state'])
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = renderer.vis(batch, model)
            end.record()
            end.synchronize()
            total_time += start.elapsed_time(end)
        render = output['render'][0]
        if render_type == 'depth':
            depth = output['depth'][0]
            depth = (depth - depth_min)/(depth_max - depth_min)
            vis = renderer.marigold_depth_vis(depth)
        elif render_type == 'height':
            depth = output['height'][0]
            print(depth.min(), depth.max(), depth.mean())
            depth = (depth - height_min)/(height_max - height_min)
            vis = renderer.marigold_depth_vis(depth)        
        else:
            vis = renderer.tensor_to_bgr(render)
        outname = os.path.join(cfg.exp, cfg.split, render_type, f'{batch["index"].item():06d}.jpg')
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, vis)
        if 'mask' in output:
            mask = output['mask'][0].detach().cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            vis = np.dstack([vis, mask[:, :, None]])
            rgbaname = os.path.join(cfg.exp, cfg.split, 'rgba', f'{batch["index"].item():06d}.png')
            os.makedirs(os.path.dirname(rgbaname), exist_ok=True)
            cv2.imwrite(rgbaname, vis)
    print('Average time: {:.2f} ms, fps: {:.1f}'.format(total_time / len(dataloader), 1000 / (total_time / len(dataloader))))
    renderer.make_video(os.path.dirname(outname), fps=cfg[cfg.split].get('fps', 30))

def validate_for_metric(exp, dataset, model, renderer, device):
    renderer.to(device)
    model.to(device)
    model.eval()
    from LoG.utils.trainer import prepare_batch
    for scale in [8, 4, 2, 1]:
        if scale not in dataset.scales: continue
        dataset.set_state(scale=scale)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        outdir = join(exp, 'test', 'scale_{}'.format(scale))
        os.makedirs(join(outdir, 'gt'), exist_ok=True)
        os.makedirs(join(outdir, 'renders'), exist_ok=True)
        total_time = 0
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch = prepare_batch(batch, device)
            with torch.no_grad():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                output = renderer.vis(batch, model)
                torch.cuda.synchronize()
                end.record()
            total_time += start.elapsed_time(end)
            append_mask = False
            # use foreground mask
            if 'mask' in batch.keys():
                mask = batch['mask'][0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                append_mask = True
            if torch.is_tensor(batch['image'][0]):
                gt = batch['image'][0].cpu().numpy()
                gt = (gt[:,:,::-1] * 255).astype(np.uint8)
                if append_mask:
                    gt = np.dstack([gt, mask[:, :, None]])
                gt_name = join(outdir, 'gt', '%04d.png'%(batch_idx))
                cv2.imwrite(gt_name, gt)
            renders = output['render'][0].permute(1, 2, 0).cpu().numpy()
            renders = (np.clip(renders[:, :,::-1], 0., 1.) * 255).astype(np.uint8)
            if append_mask:
                renders = np.dstack([renders, mask[:, :, None]])
            render_name = join(outdir, 'renders', '%04d.png'%(batch_idx))
            cv2.imwrite(render_name, renders)
        print('scale: {}, Average time: {:.2f} ms, fps: {:.1f}'.format(scale, total_time / len(dataloader), 1000 / (total_time / len(dataloader))))

def main():
    usage = 'run'
    args, cfg = Config.load_args(usage=usage)
    cfg = update_global_variable(cfg, cfg)

    exp = cfg.exp
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    print(f'Using GPUs: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    print('Write to {}'.format(exp))
    # write the parameter to the exp
    os.makedirs(exp, exist_ok=True)
    if cfg.split == 'train':
        print(cfg, file=open(os.path.join(exp, 'config.yaml'), 'w'))
    from LoG.utils.trainer import Trainer, seed_everything
    seed_everything(666)

    device = torch.device('cuda')
    model = load_object(cfg.model.module, cfg.model.args)
    if cfg.split == 'train':
        outdir = copy_git_tracked_files('./', exp)
        dataset = load_object(cfg.train.dataset.module, cfg.train.dataset.args)
        if 'base_iter' in cfg:
            base_iter = cfg.base_iter
        else:
            # round 100 iteration
            if len(dataset) < 1000:
                base_iter = (len(dataset) // 100 + 1) * 100
            else:
                base_iter = (len(dataset) // 1000 + 1) * 1000
        print('Base iteration: {}'.format(base_iter))
        model.base_iter = base_iter
        renderer = load_object(cfg.train.render.module, cfg.train.render.args)
        trainer = Trainer(cfg, model, renderer, logdir=outdir)
        trainer.to(device)
        trainer.init(dataset)
        trainer.fit(dataset)
    elif cfg.split.startswith('demo') or cfg.split == 'trainvis':
        if cfg.split == 'trainvis':
            cfg.split = 'train'
        if 'ckptname' in cfg.keys():
            model.load_state_dict(load_statedict(cfg.ckptname))
        demo(cfg, model, device)
    elif cfg.split == 'val':
        if 'ckptname' in cfg.keys():
            model.load_state_dict(load_statedict(cfg.ckptname))
        model.set_state(**cfg.val['model_state'])
        dataset = load_object(cfg.val.dataset.module, cfg.val.dataset.args)
        renderer = load_object(cfg.train.render.module, cfg.train.render.args)
        renderer.split = 'val'
        validate_for_metric(exp, dataset, model, renderer, device)

if __name__ == '__main__':
    main()
