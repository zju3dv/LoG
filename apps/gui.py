from LoG.render.renderer import NaiveRendererAndLoss
from LoG.utils.easyvolcap_utils import Viewer
from easyvolcap.engine import args
import torch
from LoG.utils.config import load_object, Config
from LoG.utils.command import update_global_variable, load_statedict, copy_git_tracked_files
from easyvolcap.utils.console_utils import catch_throw

def load_gs():
    torch.random.manual_seed(0)
    device = torch.device('cuda:0')
    # renderer
    renderer = NaiveRendererAndLoss(use_origin_render=False, background = [1., 1., 1.])
    renderer.to(device)
    filename = args.opts['filename']
    ckptname = args.opts['ckptname']
    model_cfg = Config.load(filename)
    model_cfg = update_global_variable(model_cfg, model_cfg)
    model = load_object(model_cfg.model.module, model_cfg.model.args)
    print('Load model from: ', ckptname)
    state_dict = torch.load(ckptname, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.set_state(enable_sh=True)
    model.to(device)
    model.eval()
    return model, renderer, None

if __name__ == '__main__':
    string =  '{"H":1080,"W":1920,"K":[[2139.83251953125,0.0,960.0],[0.0,2139.83251953125,496.2210388183594],[0.0,0.0,1.0]],"R":[[0.3830258846282959,0.9237375855445862,0.0],[-0.617131233215332,0.25589218735694885,0.7440917491912842],[0.687345564365387,-0.28500640392303467,0.6680806875228882]],"T":[[1.5715787410736084],[-3.9151883125305176],[5.792508125305176]],"n":0.10000000149011612,"f":1000.0,"t":0.0,"v":0.0,"bounds":[[-10.0,-10.0,-3.0],[10.0,10.0,4.0]],"mass":0.10000000149011612,"moment_of_inertia":0.10000000149011612,"movement_force":1.0,"movement_torque":1.0,"movement_speed":5.0,"origin":[0.0,0.0,0.0],"world_up":[0.0,0.0,-1.0]}'
    viewer = Viewer(camera_cfg={'type':'Camera', 'string': string})
    viewer.model, viewer.renderer, viewer.dataloader = load_gs()
    catch_throw(viewer.run)()
