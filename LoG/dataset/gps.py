import xml.dom.minidom as minidom
import os
from LoG.trajectory.gps_util import cal_distance_GPS

class GPS_Position:
    def __init__(self, lat, lon, alt):
        """
        初始化位置对象。

        参数:
        lat -- 纬度
        lon -- 经度
        alt -- 高度
        """
        self.lat = lon
        self.lon = lat  
        self.alt = alt
    
    def __sub__(self,others):
        """
        计算两个位置之间的距离。

        参数:
        others -- 另一个位置

        返回:
        两个位置之间的距离
        单位是,包含z轴距离的计算
        """
        return cal_distance_GPS(self,others)

    def get_data(self):
        """
        获取位置数据。

        返回:
        位置数据
        """
        return [self.lon, self.lat, self.alt]

    @staticmethod
    def cal_distance_np(pos1,pos2):
        # input np array
        gps1=GPS_Position(pos1[0],pos1[1],pos1[2])
        gps2=GPS_Position(pos2[0],pos2[1],pos2[2])
        return gps1-gps2

class GPS_dataset():
    
    def __init__(self,path,filename):
        """
        初始化相机位置类。

        参数:
        file_path -- XML文件路径
        """
        GPS_filename='GPS.txt'
        if not os.path.exists(os.path.join(path,GPS_filename)):
            try:
                self.positions = self.read_from_blockexchangefile(os.path.join(path,filename))
            except Exception as e:
                print(f"Error reading XML file: {e}")
                return False
            self.write_cache(os.path.join(path,GPS_filename))
        else:
            self.positions = self.read_from_cache(os.path.join(path,GPS_filename))    
    
    
    def read_from_blockexchangefile(self,xml_path):
        """
        从XML文件中读取并解析相机位置。

        参数:
        xml_path -- XML文件路径

        返回:
        一个包含相机位置的列表
        """
        with open(xml_path,'r',encoding='utf-8'):
            doc=minidom.parse(xml_path)
            root_node=doc.documentElement.getElementsByTagName('Photogroup')
            photo_node=root_node[0].getElementsByTagName('Photo')
            positions=[]
            for node in photo_node:
                GPS_node=node.getElementsByTagName('GPS')
                lon=float(GPS_node[0].getElementsByTagName('Longitude')[0].firstChild.data)
                lat=float(GPS_node[0].getElementsByTagName('Latitude')[0].firstChild.data)
                alt=float(GPS_node[0].getElementsByTagName('Altitude')[0].firstChild.data)
                positions.append(GPS_Position(lon,lat,alt))
        return positions
    
    def read_from_cache(self,cache_path):
        """
        从缓存文件中读取相机位置。

        参数:
        cache_path -- 缓存文件路径

        返回:
        一个包含相机位置的列表
        """
        positions = []
        with open(cache_path,'r',encoding='utf-8') as f:
            for line in f:
                lon, lat, alt = map(float, line.strip().split(','))
                positions.append(GPS_Position(lon, lat, alt))
        return positions

    def write_cache(self,cache_path):
        """
        将相机位置写入缓存文件。

        参数:
        cache_path -- 缓存文件路径
        """
        with open(cache_path,'w',encoding='utf-8') as f:
            for position in self.positions:
                f.write(f"{position.lon},{position.lat},{position.alt}\n")


    

if __name__ == '__main__':
    file_path='/root/repository/project/sci/log/LoG/data/Yingrenshi'
    file_name='camera_block.xml'

    camera_pos=GPS_dataset(file_path,file_name)
    print (len(camera_pos.positions))


