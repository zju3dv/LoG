import os
from os.path import join
import numpy as np
from tqdm import tqdm
from LoG.utils.config import load_object, Config
from LoG.utils.command import update_global_variable, load_statedict, copy_git_tracked_files
from LoG.trajectory.trajectory import Trajectory
from LoG.trajectory.camera_view import Camera_view
from LoG.trajectory.utils import GPU_to_colmap,colmap_gen_R_xoy
from LoG.dataset.camera_utils import get_colmap_transform
from LoG.dataset.colmap import batch_transform
from LoG.analysis.analyze import histogram_analysis
import cv2
import torch
import wandb

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

def renderability(exp, dataset, model, renderer,trajectory, device):
    renderer.to(device)
    model.to(device)
    # trajectory.to(device)
    model.eval()
    model.training=True
    from LoG.utils.trainer import prepare_batch
    scale =1 
    
    dataset.set_state(scale=scale)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    outdir = join(exp, 'test', 'scale_{}'.format(scale))
    os.makedirs(join(outdir, 'gt'), exist_ok=True)
    os.makedirs(join(outdir, 'renders'), exist_ok=True)
    os.makedirs(join(outdir, 'render_points'), exist_ok=True)
    os.makedirs(join(outdir, 'target'), exist_ok=True)
    total_time = 0

    
    # 新的视角生成
    # # 内参使用第一帧数据
    # origin_intrinsic=dataset.get_origin_intrinsic_feature()


    # #将给定的新的GPS坐标转换成colmap坐标
    # # R0_colmap,T0_colmap=dataset.get_colmap_transform()
    # R0_colmap,T0_colmap=dataset.get_colmap_transform()
    # center0=dataset.infos[0]['camera']['center']
    # R0=dataset.infos[0]['camera']['R']
    # T0=dataset.infos[0]['camera']['T']
    # #暂时使用p2p测试
    # GPS_target0,GPS_target1=trajectory.gen_camera_view()
    # #暂时规定两个视角高度均为38.0
    # GPS_target0[2]=GPS_target1[2]=38.0
    # # bugs in the transformation
    # colmap_target0=GPU_to_colmap(GPS_target0,R0_colmap,T0_colmap)
    # colmap_target1=GPU_to_colmap(GPS_target1,R0_colmap,T0_colmap)
    # target_center=colmap_target0
    # z_axis_vec=(colmap_target1-colmap_target0)[0]
    # # R=np.dot(R0,colmap_gen_R_xoy(z_axis_vec))
    # # T=-np.dot(R,target_center.T)+T0


    # # #暂时使用原始内参进行测试
    # # view_test=dataset.infos[0]
    # # view_test['camera']['R']=np.ones(3)
    # # view_test['camera']['T']=np.zeros(3)
    

    
    # R=np.eye(3)
    # T=np.dot(R,dataset.infos[0]['camera']['center'])
    # target_center=dataset.infos[0]['camera']['center']
    # view = Camera_view.generate_from_coordinate(target_center,R,T,input_intri_dict=True,intrinsic_feature=origin_intrinsic)
    # view_camera_feature=view.get_camera_feature_torch()


    '''
    在这里给出render ability的逻辑
    dataset= init()
    trajectory=init()

    # range with the length of required points num
    for i in tqdm(range(trajectory.get_render_points_num())): 
        view_selection_list=trajectory.gen_selection_view(i)
        views=[]
        n_value=1
        for view in view_selection_list:
            views.append({
                    'camera':view,
                    'value_list':torch.zeros([n_value])
                }
            )
        value_list=torch.zeros([len(selection_view_list),1]) 
        for view_index,view in enumerate(views):
            camera_feature=view['camera']
            #这里是生成一个类似与dataset的新的变量，包含了一个子集
            source_view_selection=dataset.select_view(camera_feature)
            source_dataloader = torch.utils.data.DataLoader(source_view_selection, batch_size=1, shuffle=False, num_workers=0)
            target_batch=batch_transform(camera_feature)
            #render
            target_render=renderer.vis(target_batch,model)

            source_view_num=len(source_dataset.infos)
            loss=0.0f
            for batch_idx, batch in enumerate(source_dataloader):
                # loss 放到model 中计算
                output = renderer.vis(batch, model)
                lossi+=loss_function()
            value_list[view_index]=lossi/source_view_num
        sort(view_selection_list)
        trajectory.view()
    '''


    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # batch_transformed=prepare_batch(view_camera_feature, device)
        batch_source = prepare_batch(batch, device)
        print('gen_batch')
        # pass
        with torch.no_grad():
            torch.cuda.synchronize()
            # output = renderer.vis(batch_source, model)
            output = renderer.sample(batch_source, model)
            torch.cuda.synchronize()
        append_mask = False

        # range_size analysis
        np_range_size=output['range_size'][0].cpu().numpy().reshape(-1,)
        histogram_analysis(np_range_size,'range_size',batch_idx)

        # get the max depth
        fatherest_depth_index=output['ranges'][0].cpu().numpy()[:,:,1].reshape(-1,)-1
        #use fatherest_depth_index as index to access the output['depth'][0]
        depth = output['point_depth'][0].cpu().numpy()
        depth = depth.reshape(-1)
        selected_depth = depth[fatherest_depth_index]
        histogram_analysis(selected_depth,'max_depth',batch_idx)

        #analysis the depth
        np_range_size=output['point_depth'][0].cpu().numpy().reshape(-1,)
        histogram_analysis(np_range_size,'depth',batch_idx)





        











        # use foreground mask
        if 'mask' in batch_source.keys():
            mask = batch_source['mask'][0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            append_mask = True
        if torch.is_tensor(batch['image'][0]):
            gt = batch_source['image'][0].cpu().numpy()
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

        reders_points=renderer.marigold_depth_vis(output['point_weight_pixel'][0])
        render_points_name = join(outdir, 'render_points', '%04d.png'%(batch_idx))
        cv2.imwrite(render_points_name, reders_points)




def main():
    #wandb
    # wandb.init(
    # # set the wandb project where this run will be logged
    #     project="my-awesome-project",

    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": 0,
    #         "architecture": "log",
    #         "dataset": "Lihu-domi_lowres",
    #         "epochs": 0,
    #     }
    # )
    usage = 'run'
    args, cfg = Config.load_args(usage=usage)
    cfg = update_global_variable(cfg, cfg)

    exp = cfg.exp
    # if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # for cuda debug
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
        print("useddevice:",device)
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
    elif cfg.split == 'renderability':
        if 'ckptname' in cfg.keys():
            model.load_state_dict(load_statedict(cfg.ckptname))
        model.set_state(**cfg.renderability['model_state'])
        # colmap 数据
        dataset = load_object(cfg.renderability.dataset.module, cfg.renderability.dataset.args)
        # 轨迹数据，这里暂时使用一个具有起点和终点，其逻辑为生成轨迹点，对每一个轨迹点生成视角选择集并进行打分处理
        test_id=100
        init_point=[dataset.infos[0]['gps'].get_data(),dataset.infos[test_id]['gps'].get_data()]
        trajectory=Trajectory(init_point,path_type='p2p')

        renderer = load_object(cfg.train.render.module, cfg.train.render.args)
        renderer.split = 'renderability'
        renderability(exp, dataset, model, renderer,trajectory, device)



if __name__ == '__main__':
    main()
