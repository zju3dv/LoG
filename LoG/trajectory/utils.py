import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot
from typing import List

def getProjectionMatrix2(K, H, W, znear, zfar):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]

    P = np.zeros((4, 4), dtype=np.float32)

    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = z_sign

    return P


def get_w2c_matrix(O1,R1=np.array([0,0,1])):
    '''
    默认观察视角高于xoy平面
    O1,O2分别为两个坐标系在世界坐标系原点的位置
    按照colmap设定
    R@P_c+T=P_w
    '''
    z_axis=-O1
    z_axis=z_axis/np.linalg.norm(z_axis)

    x_axis=np.cross(np.array([0,0,1]),z_axis)
    x_axis=x_axis/np.linalg.norm(x_axis)

    y_axis=np.cross(x_axis,z_axis)
    y_axis=y_axis/np.linalg.norm(y_axis)

    R=np.vstack(np.array([x_axis,y_axis,z_axis])).T
    chect=np.dot(R,R.T)
    T=np.dot(-R.T,O1)
    w2c_matrix=np.hstack((R,T.reshape(3,1)))

    return w2c_matrix

def get_w2c_matrix_y_z(O1,degree=45):
    '''
    生成一个从这个坐标向y轴正方向与z轴负方向成45°的视角
    '''
    dir_vec=np.array([0.,1.,0.])
    up_vec = np.array([0, 0, -1])

    alpha=math.radians(degree)
    x_axis=np.array([-1,0,0])
    z_axix=np.array([0,1.*math.cos(alpha),-1.*math.sin(alpha)])
    y_axis=np.cross(x_axis,z_axix)

    R=np.vstack(np.array([x_axis,y_axis,z_axix])).T
    T = np.dot(-R.T, O1)
    w2c_matrix=np.hstack((R,T.reshape(3,1)))
    return w2c_matrix

def gen_R_xoy(z_axis):
    '''
    input numpy array

    给定zaxis生成x轴与xoy平面平行的坐标系

    采用世界标准坐标系,即x向右,y向内,z向上
    '''
    z_axis=z_axis/np.linalg.norm(z_axis)
    if(z_axis[2]>0):
        x_axis=np.cross(np.array([0.,0.,1.]),z_axis)
        x_axis=x_axis/np.linalg.norm(x_axis)
    else:
        x_axis=np.cross(z_axis,np.array([0.,0.,-1.]))
        x_axis=x_axis/np.linalg.norm(x_axis)
    y_axis=np.cross(x_axis,z_axis)
    y_axis=y_axis/np.linalg.norm(y_axis)

    return np.vstack(np.array([x_axis,y_axis,z_axis])).T

def colmap_gen_R_xoy(z_axis):
    '''
    input numpy array

    给定zaxis生成x轴与xoy平面平行的坐标系

    采用colmap系,即x向右,y向下,z向内
    '''
    z_axis=z_axis/np.linalg.norm(z_axis)
    if(z_axis[2]>0):
        x_axis=np.cross(np.array(z_axis,[0.,-1.,0.]))
        x_axis=x_axis/np.linalg.norm(x_axis)
    else:
        x_axis=np.cross(np.array([0.,1.,0.],z_axis))
        x_axis=x_axis/np.linalg.norm(x_axis)
    y_axis=np.cross(x_axis,z_axis)
    y_axis=y_axis/np.linalg.norm(y_axis)

    return np.vstack(np.array([x_axis,y_axis,z_axis])).T

def cal_rotate_matrix(a,theta):
    """
    a:旋转轴
    theta:旋转角度
    这里生成的旋转矩阵对应规则为 R@P_camera=P_world
    """
    mrp=a*np.tan(theta/4)
    rotation_matrix=Rot.from_mrp(mrp).as_matrix()
    return rotation_matrix

def cal_distance_GPS(position1,position2):
        """
        暂时depricate 已在GPS_position实现
        计算两个位置之间的距离。

        参数:
        position1 -- 位置1
        position2 -- 位置2

        返回:
        两个位置之间的距离
        """
        from math import radians, cos, sin, asin, sqrt

        # 将经纬度转换为弧度
        lat1, lon1, lat2, lon2 = map(radians, [position1.lat, position1.lon, position2.lat, position2.lon])
        z1,z2=position1.alt,position2.alt
        # haversine公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371.0  # 地球平均半径，单位为公里
        dis_xoy=c * r * 1000.
        dis=sqrt(dis_xoy**2+(z1-z2)**2)
        return  dis

def GPU_to_colmap(gps_pos:List[float],R,T):
    '''
    
    给定wgs84gps坐标,colmap第一帧图像的R,T
    计算gps坐标在colmap坐标
    需要减去第一帧的ecef坐标

    '''
    import pyproj

    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    wgs84 = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    # transform wgs84 to ecef
    x, y, z = pyproj.transform(wgs84, ecef, gps_pos[1], gps_pos[0], gps_pos[2],radians=False)
    temp=np.dot(R, np.array([x, y, z]))
    camera_center = np.dot(R, np.array([x, y, z])) + T
    return camera_center



# def 

if __name__=="__main__":
    # O1=np.array([0.,0.,0.])
    # w2c_matrix=get_w2c_matrix_y_z(O1)
    # R=w2c_matrix[:,:3]
    # T=w2c_matrix[:,3]
    # c1=np.dot(R.T,np.array([1.,1.,0.]))+T
    # c2=np.dot(R.T,np.array([0.,0.,0.]))+T
    # c3=np.dot(R.T,np.array([2.,0.,0.]))+T
    # print(w2c_matrix)

    # P1=np.array([0.70710678 ,0.70710678 ,0.         ,1.        ])
    # a=np.array([1.,0.,0.])
    # theta=math.radians(90)
    # R=cal_rotate_matrix(a,theta)
    # print(R)
    # print(np.dot(R.T,P1))

    # from LoG.dataset.gps import GPS_Position
    # p1=GPS_Position(22.64128804,113.92305221,120.04)
    # p2=GPS_Position(22.64154932,113.92420099,38.7)

    # print(cal_distance_GPS(p1,p2))

    z_axis= np.array([1.,1.,1.])
    R=gen_R_xoy(z_axis)
    print(R)




