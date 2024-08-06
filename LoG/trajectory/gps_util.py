import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot
from typing import List

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
        lon1, lat1, lon2,  lat2 = map(radians, [position1.lon, position1.lat, position2.lon, position2.lat])
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