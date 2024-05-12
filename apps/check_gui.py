from LoG.render.renderer import NaiveRendererAndLoss
import torch
from LoG.dataset.demo import DemoDataset
from LoG.utils.easyvolcap_utils import Viewer
from easyvolcap.utils.console_utils import catch_throw

def init_random_gs(num_points=1000):
    torch.random.manual_seed(0)
    device = torch.device('cuda:0')
    record = {
        'xyz': torch.rand(num_points, 3) - 0.5,
        'scaling': torch.rand(num_points, 3) * 0.05,
        'opacity': torch.ones(num_points, 1) * 0.999,
        'colors': torch.rand(num_points, 3),
        'rotation': torch.rand(num_points, 4),
    }
    dataset = DemoDataset(size=2048, ranges=[0, 360, 300])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    renderer = NaiveRendererAndLoss(use_origin_render=True, background = [1., 1., 1.])
    renderer.to(device)

    num_points = record['xyz'].shape[0]
    from LoG.model.base_gaussian import BaseGaussian
    model = BaseGaussian.create_from_record(record)
    model.to(device)

    return model, renderer, dataloader

if __name__ == '__main__':
    string =  '{"H":1080,"W":1920,"K":[[2139,0.0,960.0],[0.0,2139,540],[0.0,0.0,1.0]],"R":[[0.0,1,0.0],[0,0.,1],[1,0,0]],"T":[[0],[0],[5.]],"n":0.1,"f":1000.0,"t":0.0,"v":0.0,"bounds":[[-10.0,-10.0,-3.0],[10.0,10.0,4.0]],"mass":0.1,"moment_of_inertia":0.1,"movement_force":1.0,"movement_torque":1.0,"movement_speed":5.0,"origin":[0.0,0.0,0.0],"world_up":[0.0,0.0,-1.0]}'
    viewer = Viewer(camera_cfg={'type':'Camera', 'string': string})
    viewer.model, viewer.renderer, viewer.dataloader = init_random_gs()
    catch_throw(viewer.run)()
