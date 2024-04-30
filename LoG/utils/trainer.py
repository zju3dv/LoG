import os
from os.path import join
import time
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from .recorder import Recorder
from .config import load_object, Config
from collections import defaultdict
# import matplotlib.pyplot as plt

def imwrite(imgname, img):
    os.makedirs(os.path.dirname(imgname), exist_ok=True)
    cv2.imwrite(imgname, img)

def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_batch(data, device):
    batch = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            batch[key] = torch.FloatTensor(val).to(device)
        elif key == 'camera' or key == 'camera_depth':
            for camk in val.keys():
                if isinstance(val[camk], np.ndarray):
                    val[camk] = torch.FloatTensor(val[camk]).to(device)
                elif torch.is_tensor(val[camk]):
                    val[camk] = val[camk].float().to(device)
                else:
                    import ipdb; ipdb.set_trace()
            batch[key] = val
        elif torch.is_tensor(val):
            batch[key] = val.to(device)
        else:
            batch[key] = val
    return batch

class Trainer(nn.Module):
    def __init__(self, cfg, model, render, logdir='log'):
        super().__init__()
        self.cfg = cfg
        self.exp = cfg.exp
        os.makedirs(self.exp, exist_ok=True)
        self.model = model
        self.render = render
        self.recorder = Recorder(logdir)
        self.check_val()
        self.check_overlook()
        self.log_inverval = cfg.get('log_interval', 1000)
        self.save_interval = cfg.get('save_interval', 100_000)
    
    def check_val(self):
        if 'val' not in self.cfg:
            self.val = None
        else:
            dataset = load_object(self.cfg.val.dataset.module, self.cfg.val.dataset.args)
            print(f'>>> Load val dataset: {len(dataset)}')
            val_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                drop_last=False,
            )
            self.val = val_loader
            if 'render' in self.cfg.val:
                self.render_val = load_object(self.cfg.val.render.module, self.cfg.val.render.args)
            else:
                self.render_val = self.render
            if dataset.scales[0] < 4:
                self.lpips = None
            else:
                import lpips
                self.lpips = lpips.LPIPS(net='vgg', spatial=False)

    def check_overlook(self):
        if 'overlook' not in self.cfg:
            self.overlook = None
        else:
            dataset = load_object(self.cfg.overlook.dataset.module, self.cfg.overlook.dataset.args)
            print(f'>>> Load overlook dataset: {len(dataset)}')
            val_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=4,
                shuffle=False,
                drop_last=False,
            )
            self.overlook = val_loader
        if 'overlook_oneframe' not in self.cfg:
            self.overlook_oneframe = None
        else:
            dataset = load_object(self.cfg.overlook_oneframe.dataset.module, self.cfg.overlook_oneframe.dataset.args)
            print(f'>>> Load overlook_oneframe dataset: {len(dataset)}')
            self.overlook_oneframe = dataset
            self.overlook_oneframe_freq = self.cfg.overlook_oneframe.iteration

    def to(self, device):
        self.device = device
        return super().to(device)

    def train_loader(self, dataset, args=None, base_iter=1):
        if args is None:
            stage = self.cfg.train.loader.args
        else:
            stage = args
        batch_size = stage.get('batch_size', 16)
        iterations = stage.get('iterations', 1024) * base_iter
        num_workers = stage.get('num_workers', 8)
        def worker_init_fn(worker_id):
            np.random.seed(worker_id + 42)
        from .sampler import IterationBasedSampler
        trainloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=IterationBasedSampler(dataset, iterations * batch_size),
            drop_last=True,
            worker_init_fn=worker_init_fn
        )
        return trainloader
    
    def val_loader(self, dataset, index=None, num_workers=1):
        from .sampler import IndexSampler
        val_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=IndexSampler(dataset, index),
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )
        return val_loader

    def training_step(self, model, data, step=True, accumulate_step=1):
        batch = prepare_batch(data, self.device)
        output = self.render(batch, model)
        # check the visible points
        if 'index' in output['visibility_flag'][0].keys() and output['visibility_flag'][0]['index'].shape[0] == 0:
            if 'index_node' in output['visibility_flag'][0].keys():
                pass
            else:
                return False, {}, 0.
        if 'point_id' in output.keys():
            if output['point_id'][0].shape[0] == 0:
                print(f'visible points: {output["point_id"][0].shape[0]}')
                return False, {}, 0.
        loss = output['loss'] / accumulate_step
        loss.backward()
        model.update_by_output(output)
        model.step()
        if self.global_iterations % 10 == 0:
            for key, val in output['loss_dict'].items():
                self.recorder.log(self.global_iterations, f'train/loss_{key}', val)
            self.recorder.log(self.global_iterations, 'train/loss', loss)
        return True, output, loss.item()

    def init(self, dataset):
        dataset.read_img = False
        os.makedirs(join(self.exp, 'init'), exist_ok=True)
        if 'init' in self.cfg.train:
            dataset.set_state(**self.cfg.train.init.dataset_state)
            valloader = self.val_loader(dataset, num_workers=0)
            self.model.at_init_start()
            for iteration, data in enumerate(tqdm(valloader, desc='initialize the model')):
                # timer the process
                self.model.clear()
                batch = prepare_batch(data, self.device)
                self.model.init(self.render, batch, iteration)
            self.model.at_init_final()
            partial_indices = list(range(len(dataset)))
        else:
            partial_indices = list(range(len(dataset)))
        dataset.set_partial_indices(partial_indices)
        self.model.eval()
        quickview = self.val_loader(dataset)
        for iteration, data in enumerate(tqdm(quickview, desc='quick view')):
            batch = prepare_batch(data, self.device)
            ret = self.render.vis(batch, self.model)
            vis = ret['render'][0]
            vis = self.render.tensor_to_bgr(vis)
            img = cv2.imread(batch['imgname'][0])
            vis = np.vstack([vis, img])
            cv2.imwrite(join(self.exp, 'init', f'model_{iteration}_{os.path.basename(batch["imgname"][0])}.jpg'), vis)
            if iteration > 10:
                break
        self.model.train()
        dataset.read_img = True
        dataset.set_partial_indices(partial_indices)
        if self.overlook is not None:
            self.model.eval()
            with torch.no_grad():
                for iteration, data in enumerate(tqdm(self.overlook)):
                    batch = prepare_batch(data, self.device)
                    self.model.clear()
                    ret = self.render.vis(batch, self.model)
                    vis = ret['render'][0]
                    vis = self.render.tensor_to_bgr(vis)
                    cv2.imwrite(join(self.exp, f'overlook_{iteration}.jpg'), vis)
            self.model.train()

    def log_radius(self, output):
        radii = output['radii'][0].detach().cpu().numpy()
        self.recorder.writer.add_histogram('train/radius3d', radii, self.global_iterations, bins=100)
        # log grad
        grad = output['viewspace_points'][0].grad
        grad_norm = torch.norm(grad, dim=-1).cpu().numpy() * 1e4
        grad_norm = grad_norm[grad_norm > 1e-7]
        self.recorder.writer.add_histogram('train/grad_norm', grad_norm, self.global_iterations, bins=100)

    def log_opacity(self, output):
        opacity = output['opacity'][0][:, 0].detach().cpu().numpy()
        self.recorder.writer.add_histogram('train/opacity', opacity, self.global_iterations, bins=100)

    def log_point_cloud(self, output):
        from .file import write_ply
        num_points = [i.shape[0] for i in output['xyz']]
        num_points = sum(num_points) / len(num_points)
        self.recorder.log(self.global_iterations, 'train/visible_points', num_points)
        xyz = output['xyz'][0].detach()
        colors = output['colors'][0].detach()
        outname = os.path.join(self.exp, 'pointcloud', '{:06d}.ply'.format(self.global_iterations))
        write_ply(outname, xyz.cpu().numpy(), colors.cpu().numpy())

    def log_gpu(self):
        self.recorder.log(self.global_iterations, 'train/memory', torch.cuda.memory_allocated() / 2**20)
        self.recorder.log(self.global_iterations, 'train/max_mem', torch.cuda.max_memory_allocated() / 2**20)
        self.recorder.log(self.global_iterations, 'train/num_points', self.model.num_points)

    def log_grad(self, data):
        batch = prepare_batch(data, self.device)
        output = self.render.vis_grad(batch, self.model)
        render = output['render'][0].detach()
        vis_grad_max = 5e-3
        grad_acc = torch.clamp(render[0], 0., vis_grad_max) / vis_grad_max
        vis_grad = self.render.marigold_depth_vis(grad_acc)
        cv2.putText(vis_grad, f'{render[0].min().item():.4f}->{render[0].max().item():.4f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        outname = os.path.join(self.exp, 'grad', '{:06d}.jpg'.format(self.global_iterations))
        imwrite(outname, vis_grad)

    @torch.no_grad()
    def log_in_training(self, batch_idx, batch_total, data, loss, output, vis_residual=False):
        global_time = time.time() - self.global_start_time
        self.recorder.log(self.global_iterations, 'train/time', global_time)
        current_time = time.time() - self.start_time
        print(f'[{self.global_iterations:6d}: {batch_idx:6d}/{batch_total:6d}] {current_time:4.1f}s loss: {loss:.4f} model {self.model}')
        self.start_time = time.time()
        vis_list = []
        for key in ['gt', 'render', 'render_correct', 'render_max', 'render_id', 'render_root', 'render_leaf']:
            if key in output.keys():
                vis_tensor = output[key][0].detach()
                vis_numpy = self.render.tensor_to_bgr(vis_tensor)
                cv2.putText(vis_numpy, f'{key}', (10, vis_numpy.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                vis_list.append(vis_numpy)
        vis = np.hstack(vis_list)
        if 'point_weight_pixel' in output.keys():
            vis_ = self.render.marigold_depth_vis(output['point_weight_pixel'][0])
            vis = np.hstack([vis, vis_])
        cv2.putText(vis, f'{data["imgname"][0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        outname = os.path.join(self.exp, 'vis', '{:06d}.jpg'.format(self.global_iterations))
        imwrite(outname, vis)
        if vis_residual:
            residual = torch.norm(output['render'][0] - output['gt'][0], dim=0)
            residual = torch.clamp(residual, min=0., max=1.)
            vis_residual = self.render.marigold_depth_vis(residual)
            outname = os.path.join(self.exp, 'residual', '{:06d}.jpg'.format(self.global_iterations))
            imwrite(outname, vis_residual)
        # vis depth
        if 'gt_depth' in output.keys():
            depth = torch.cat([output['gt_depth'][0], output['depth_ssi_map'][0].detach()], dim=0)
            vis_depth = self.render.marigold_depth_vis(depth)
            outname = os.path.join(self.exp, 'depth', '{:06d}.jpg'.format(self.global_iterations))
            imwrite(outname, vis_depth)
            # write the height map
            if 'height_map' in output:
                vis_height = output['height_map'][0].detach().cpu()
                vis_height = (vis_height - vis_height.min()) / (vis_height.max() - vis_height.min())
                vis_height = self.render.marigold_depth_vis(vis_height)
                outname = os.path.join(self.exp, 'height', '{:06d}.jpg'.format(self.global_iterations))
                imwrite(outname, vis_height)
        if 'acc_map' in output.keys():
            acc = output['acc_map'][0]
            vis_acc = self.render.marigold_depth_vis(acc)
            outname = os.path.join(self.exp, 'acc', '{:06d}.jpg'.format(self.global_iterations))
            imwrite(outname, vis_acc)
        # self.log_radius(output)
        # self.log_opacity(output)
        if self.cfg.get('log_pointcloud', False):
            try:
                self.log_point_cloud(output)
            except:
                print('log point cloud failed')
        self.log_gpu()
        # self.log_grad(data)

    @torch.no_grad()
    def make_validation(self, iteration, visualize=False):
        from .metric import psnr
        metric = defaultdict(list)
        model = self.model
        model.eval()
        logdir = os.path.join(self.exp, 'val', f'{iteration:06d}')
        for _data in tqdm(self.val, desc=f'val {iteration}'):
            batch = prepare_batch(_data, self.device)
            model.clear()
            output = self.render_val.vis(batch, self.model)
            pred = output['render'][0].detach()
            pred = self.render_val.process_pred(batch, pred)
            gt = self.render_val.process_gt(batch)[0]
            del output
            # calculate the metrics
            l1 = torch.mean(torch.abs(pred - gt))
            _psnr = psnr(pred, gt)
            metric['l1'].append(l1)
            metric['psnr'].append(_psnr)
            if self.lpips is not None:
                ret_lpips = self.lpips(pred[None], gt[None], retPerLayer=False, normalize=True)
                metric['lpips'].append(ret_lpips.item())
            metric['imgname'].append(batch['imgname'][0])
            # write images
            if (iteration + 1) % 1000 == 0 or visualize:
                os.makedirs(logdir, exist_ok=True)
                outname = join(logdir, f'{batch["index"][0]:06d}_{os.path.basename(metric["imgname"][-1])}.jpg')
                vis = self.render_val.tensor_to_bgr(torch.cat([pred, gt], dim=1))
                cv2.imwrite(outname, vis)
        # summary the metrics
        record = {
            'iteration': iteration,
            'num_points': model.num_points,
        }
        print(f'>>> Validation: {iteration}: {len(metric["imgname"])} images')
        for key, val in metric.items():
            if key == 'imgname':
                continue
            mean_val = sum(val)/len(val)
            record[key] = mean_val
            if self.global_iterations > 0:
                self.recorder.log(self.global_iterations, f'val/{key}', mean_val)
            print(f'    - {key}: {mean_val:.4f}')
        logname = os.path.join(self.exp, 'val', f'{iteration:06d}.yml')
        os.makedirs(os.path.dirname(logname), exist_ok=True)
        record['records'] = []
        for i in range(len(metric['imgname'])):
            record['records'].append({
                'imgname': metric['imgname'][i],
                'l1': metric['l1'][i],
                'psnr': metric['psnr'][i],
            })
            if 'lpips' in metric:
                record['records'][-1]['lpips'] = metric['lpips'][i]
        if False:
            import yaml
            try:
                yaml.dump(record, open(logname, 'w'))
            except:
                print('write evaluation log failed')
        self.model.train()

    @torch.no_grad()
    def make_overlook(self, mode='rgb', iteration=-1):
        if iteration == -1:
            iteration = self.global_iterations
        model = self.model
        text = ''
        if mode == 'offset':
            if getattr(model, 'offset', None) is None:
                return 0
        if mode == 'grad':
            grad_acc = self.model.get_grad_acc()
            grad_acc = torch.clamp_min(grad_acc, 1e-4) - 1e-4
            text = grad_acc.max()
            grad_acc = grad_acc / grad_acc.max()
            grad_acc = torch.stack([grad_acc, grad_acc, grad_acc], dim=1)
        elif mode == 'offset':
            offset = torch.tanh(model.offset.detach())
            offset = torch.norm(offset, dim=-1)
            print(f'offset: {offset.mean()}, max {offset.max()}')
            # offset = (torch.clamp(offset, min=0.5, max=1) - 0.5) * 2
            offset = torch.stack([offset, offset, offset], dim=1)
        self.model.eval()
        for _iter, _data in enumerate(self.overlook):
            batch = prepare_batch(_data, self.device)
            self.model.clear()
            if mode == 'rgb':
                output = self.render.vis(batch, self.model)
                vis = self.render.tensor_to_bgr(output['render'][0])
            elif mode == 'grad':
                output = self.render.vis(batch, self.model, features=grad_acc)
                vis = self.render.acc_to_bgr(output['render'][0][0])
            elif mode == 'offset':
                output = self.render.vis(batch, self.model, features=offset)
                # vis = self.render.tensor_to_bgr(output['render'][0])
                vis = self.render.acc_to_bgr(output['render'][0][0])
            cv2.putText(vis, f'{text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            del output
            outname = os.path.join(self.exp, 'overlook', '{}_{:06d}_{:02d}.jpg'.format(mode, iteration, _iter))
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            cv2.imwrite(outname, vis)
            if mode != 'rgb':
                break
        self.model.train()

    @torch.no_grad()
    def make_overlook_oneframe(self, iteration=-1):
        iteration = self.global_iterations // self.overlook_oneframe_freq
        model = self.model
        self.model.eval()
        len_data = len(self.overlook_oneframe)
        data = self.overlook_oneframe[iteration % len_data]
        batch = torch.utils.data.default_collate([data])
        batch = prepare_batch(batch, self.device)
        self.model.clear()
        output = self.render.vis(batch, model)
        vis = self.render.tensor_to_bgr(output['render'][0])
        cv2.putText(vis, f'{iteration}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        del output
        outname = os.path.join(self.exp, 'overlook_oneframe', 'rgb', '{:06d}.jpg'.format(iteration))
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, vis)
        self.model.train()

    def save_ckpt(self, ckptname):
        state_dict = self.model.state_dict()
        state_dict_wooptim = {}
        for k, v in state_dict.items():
            if 'optimizer' in k:
                continue
            if 'counter' in k:
                continue
            state_dict_wooptim[k] = v
        torch.save({
            'state_dict': state_dict,
            'global_iterations': self.global_iterations
            }, 
        ckptname)
        torch.save(state_dict_wooptim, ckptname.replace('.pth', '_wotrain.pth'))

    def check_iteration(self, stage_name, iteration, cfg_iteration):
        if cfg_iteration == -1:
            return False
        if isinstance(cfg_iteration, int) and (iteration) % cfg_iteration == 0:
            return True
        if isinstance(cfg_iteration, dict):
            if stage_name not in cfg_iteration:
                return False
            iters = cfg_iteration[stage_name]
            if iteration > iters[0] and iteration < iters[1] and iteration % iters[2] == 0:
                return True
        return False

    def fit(self, dataset):
        self.global_iterations = 0
        self.global_start_time = time.time()
        for stage_name, stage in self.cfg.train.stages.items():
            print(f'> Run stage: {stage_name}. {stage.loader.args.iterations * self.model.base_iter} iterations')
            if 'ckptname' in stage:
                ckptname = stage.ckptname
            else:
                ckptname = join(self.exp, f'model_{stage_name}.pth')
            if os.path.exists(ckptname):
                print(f'Load checkpoint: {ckptname}')
                statedict = torch.load(ckptname, map_location=self.device)
                self.model.load_state_dict(statedict['state_dict'], split='train')
                # self.model.check_num_points()
                self.global_iterations += stage.loader.args.iterations * self.model.base_iter
                continue
            dataset.set_state(**stage.dataset_state)
            self.model.set_stage(stage_name)
            self.model.set_state(**stage.model_state)
            trainloader = self.train_loader(dataset, stage.loader.args, base_iter=self.model.base_iter)
            self.recorder.log(self.global_iterations, 'train/batch_size', stage.loader.args.batch_size)
            self.model.training_setup()
            if self.val is not None:
                self.make_validation(self.global_iterations + 1)
            self.start_time = time.time()
            need_log = True
            moving_mean_loss = 0
            print(self.model)
            for iteration, data in enumerate(trainloader):
                self.model.clear()
                self.render.iteration = self.global_iterations
                flag, output, loss = self.training_step(self.model, data)
                if not flag:
                    self.global_iterations += 1
                    continue
                moving_mean_loss += loss
                if (iteration + 1) % self.log_inverval == 0 or need_log:
                    need_log = False
                    self.log_in_training(iteration, len(trainloader), data, moving_mean_loss / self.log_inverval, output)
                    if (iteration + 1) % self.log_inverval == 0 and iteration > 0:
                        self.recorder.log(self.global_iterations, 'train/loss_mean', moving_mean_loss / self.log_inverval)
                        moving_mean_loss = 0
                del output, loss
                if self.val is not None and (iteration + 1) % self.cfg.val.iteration == 0:
                    self.make_validation(self.global_iterations)
                # Do the overlook
                if self.overlook is not None and self.check_iteration(stage_name, iteration + 1, self.cfg.overlook.iteration):
                    self.make_overlook()
                if self.overlook_oneframe is not None and self.cfg.overlook_oneframe.iteration > 0 and (iteration) % self.cfg.overlook_oneframe.iteration == 0:
                    self.make_overlook_oneframe()
                if (iteration + 1) % self.save_interval == 0:
                    if True:
                        name = 'model_latest.pth'
                    else:
                        name = f'model_{self.global_iterations:06d}.pth'
                    print(f'Save checkpoint...: ', join(self.exp, name))
                    self.save_ckpt(join(self.exp, name))
                with torch.no_grad():
                    if (iteration + 1) < len(trainloader):
                        # only update during fitting process
                        flag_update = self.model.update_by_iteration(iteration, self.global_iterations)
                        if flag_update:
                            need_log = True
                            if self.overlook is not None and self.check_iteration(stage_name, iteration + 1, self.cfg.overlook.iteration):
                                self.make_overlook(iteration=self.global_iterations+1)
                            # self.make_overlook(mode='grad', iteration=self.global_iterations + 1)
                                # self.make_overlook(mode='offset', iteration=self.global_iterations + 1)
                            self.recorder.log(self.global_iterations, 'train/num_points', self.model.num_points)
                if self.global_iterations % 10 == 0:
                    self.recorder.log(self.global_iterations, 'train/lr', self.model.lr)
                self.global_iterations += 1
            ckptname = join(self.exp, f'model_{stage_name}.pth')
            self.save_ckpt(ckptname)            