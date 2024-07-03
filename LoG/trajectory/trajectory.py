import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
# from ..dataset.gps import GPS_Position
import torch

class Trajectory(torch.nn.Module):
    """
    trajectory 的坐标都是基于GPS的,如果需要计算对应的colmap坐标视角关系,需要使用camera_view的对应的方法
    points: [N,3]
    暂时设置两种逻辑
    1. 两点间轨迹生成 p2p
        start_point 起点
        dest_point 终点
    2. 单点针对某点生成固定轨迹 p-route
    """
    def __init__(self,points_data,path_type='p2p',crucial_pt_num=1,route_type=None):
        if path_type=='p2p':
            self.path_type=path_type
            self.p2p_init(points_data)
            pass
        elif path_type=='p-route':
            # 数据定义：
            # points 的前n个点 为关键点，一般为环绕中心点,一般为一个
            #剩余的所有点均为路径点，其中第一个为起点，如果route_type是slerp 则第一个点就是终点
            self.path_type=path_type
            self.route_type=route_type
            self.crucial_pt_num=crucial_pt_num
            self.p_route_init(points_data,crucial_pt_num,route_type)
            pass
        # self.init_points=points

    def set_num_points(self,num_point):
        self.num_points=num_point

    def interpolation_path(self,strategy="slerp"):
        #still have bugs
        # on working
        pointlines=[]
        if strategy=="slerp":
            for i in range(len(self.init_points)):
                st_ed_pos=[self.init_points[i],self.init_points[i+1]]
                interp_points=self.interpolate_path_slerp(st_ed_pos)

            pointlines+=self.init_points[-1]
        return pointlines
    def interpolate_path_slerp(self,st_ed_pos):
        """
        -working on it
        interpolate the trajectory using slerp
        """
        st=st_ed_pos[0]
        ed=st_ed_pos[1]
        key_rots= R.from_matrix()

    def p2p_init(self,points):

        assert len(points)>1
        self.start_point=points[0]
        self.dest_point=points[-1]
        self.points=points
    
    def gen_camera_view(self):
        v1=self.start_point
        v2=self.dest_point
        return v1,v2

    def p_route_init(self,points,crucial_pt_num=1,route_type='slerp'):
        assert len(points)>1
        self.crucial_points=points[:crucial_pt_num]
        self.route_points=points[crucial_pt_num:]
        self.points=points

    def gen_route(self):
        if self.path_type=='p-route':
            if self.route_type=='slerp':
                return self.interpolate_path_slerp()
            else:
                pass
    
    # def interpolate_path_slerp(self):
        #slerp 是针对旋转矩阵的插值



if __name__ == "__main__":
    key_rots=R.random(5,random_state=114515)
    key_times = [0, 1, 2, 3, 4]
    slerp = Slerp(key_times, key_rots)
    times = [0, 0.5, 0.25, 1, 1.5, 2, 2.75, 3, 3.25, 3.60, 4]
    interp_rots = slerp(times)
    print(key_rots.as_euler('xyz', degrees=True))
    print(interp_rots.as_euler('xyz', degrees=True))
