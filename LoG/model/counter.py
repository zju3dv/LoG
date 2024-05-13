import torch
import torch.nn as nn

class Counter(nn.Module):
    def __init__(self, num_points, add_depth=False) -> None:
        super().__init__()
        zero_float32 = torch.zeros((num_points,), dtype=torch.float32)
        zero_int16 = torch.zeros((num_points,), dtype=torch.int16)
        zero_int32 = torch.zeros((num_points,), dtype=torch.int32)
        self.register_buffer('weights_max', zero_float32.clone())
        self.register_buffer('weights_sum', zero_float32.clone())
        self.register_buffer('grad_sum', zero_float32.clone())
        for name in ['radii_max', 'visible_count']:
            self.register_buffer(name, zero_int16.clone())
        self.register_buffer('radii_max_max', zero_int32.clone())
        self.register_buffer('area_sum', zero_int32.clone())
        self.register_buffer('radius3d_min', zero_float32.clone() + 1)
        self.register_buffer('radius3d_max', zero_float32.clone() + 1)
        self.register_buffer('create_steps', zero_int32.clone())

    def get_gradmean(self):
        return self.grad_sum / torch.clamp(self.area_sum, min=1)

    def str_min_mean_max(self, name, data):
        return f'{name:10s} {data.shape[0]:8d} [{data.min().item():.5f}~{data.float().mean().item():.5f}+{data.float().std().item():.5f}~{data.max().item():.5f}]'
    
    def reset(self, num_points):
        print(f'[{self.__class__.__name__}] reset counter -> {num_points}')
        for key in ['weights_max', 'weights_sum', 'radii_max', 'radii_max_max', 'area_sum', 'grad_sum', 'visible_count']:
            data = getattr(self, key)
            data.set_(torch.zeros((num_points,), device=data.device, dtype=data.dtype))

    def reset_create_steps(self):
        self.create_steps.fill_(0)

    def update_by_output(self, output, fix_parent=False):
        for i in range(len(output['render'])):
            # read out
            visible_index = output['visibility_flag'][i]['index']
            grad = output['viewspace_points'][i].grad
            if 'index_node' in output['visibility_flag'][i]:
                visible_index = torch.cat([
                    visible_index, 
                    output['visibility_flag'][i]['index_node']])
            radii = output['radii'][i]
            grad_norm = torch.norm(grad[:, :2], dim=-1)
            weights_max = output['point_weight'][i].data
            flag_vis = radii > 0
            index_vis = torch.where(flag_vis)[0]
            output['visibility_flag'][i]['flag_vis'] = flag_vis
            output['visibility_flag'][i]['index_vis'] = index_vis
            # update the counter
            point_id = output['point_id'][i]
            # count the points hit the maximum pixel
            point_count = output['point_count'][i]
            # sum the total area
            self.area_sum[visible_index[point_id]] += point_count
            vis_visible_index = visible_index[index_vis]
            self.create_steps[vis_visible_index] += 1
            self.visible_count[vis_visible_index] += 1
            self.weights_max[vis_visible_index] = torch.max(self.weights_max[vis_visible_index], weights_max[index_vis])
            self.weights_sum[vis_visible_index] += weights_max[index_vis]
            # only count the points which is the maximum response
            # weight by the area
            self.grad_sum[visible_index[point_id]] += grad_norm[point_id] * point_count
            # self.grad_sum[vis_visible_index] += grad_norm[index_vis] * point_count
            self.radii_max[vis_visible_index] = torch.max(self.radii_max[vis_visible_index], radii[index_vis].short())
            self.radii_max_max[visible_index[point_id]] = torch.maximum(point_count.int(), self.radii_max_max[visible_index[point_id]])