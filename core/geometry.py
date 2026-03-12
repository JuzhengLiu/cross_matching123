# C:\Users\xiaoju\Desktop\eloft-pipei\cross_matching\core\geometry.py
# -*- coding: utf-8 -*-

import math
import numpy as np
from osgeo import gdal, osr

class GeoTransformer:
    def __init__(self, tif_path):
        """
        初始化：读取基准图的地理参考信息，建立经纬度与像素坐标的转换桥梁。
        """
        gdal.AllRegister()
        self.dataset = gdal.Open(tif_path)
        if self.dataset is None:
            raise FileNotFoundError(f"无法读取基准图文件: {tif_path}")
        
        # 解析 TIF 图的投影参考系（如 UTM）和地理参考系（如 WGS84）
        self.prosrs = osr.SpatialReference()
        self.prosrs.ImportFromWkt(self.dataset.GetProjection())
        self.geosrs = self.prosrs.CloneGeogCS()
        
        # 创建相互转换的坐标工具
        self.ct_geo2lonlat = osr.CoordinateTransformation(self.prosrs, self.geosrs)
        self.ct_lonlat2geo = osr.CoordinateTransformation(self.geosrs, self.prosrs)
        
        # 获取影像六参数：用于像素坐标与投影坐标（米）之间的转换
        self.transform = self.dataset.GetGeoTransform()
        self.EARTH_RADIUS = 6371393.0 # 地球半径单位米

    def pixel_to_lonlat(self, x, y):
        """将像素行列号转为 WGS84 经纬度"""
        px = self.transform[0] + x * self.transform[1] + y * self.transform[2]
        py = self.transform[3] + x * self.transform[4] + y * self.transform[5]
        coords = self.ct_geo2lonlat.TransformPoint(px, py)
        return coords[1], coords[0] # 返回经度, 纬度

    def lonlat_to_pixel(self, lon, lat):
        """将经纬度反算为基准图上的像素位置"""
        coords = self.ct_lonlat2geo.TransformPoint(lat, lon)
        px, py = coords[0], coords[1]
        a = np.array([[self.transform[1], self.transform[2]], 
                      [self.transform[4], self.transform[5]]])
        b = np.array([px - self.transform[0], py - self.transform[3]])
        pixel_coords = np.linalg.solve(a, b)
        return [pixel_coords[0], pixel_coords[1]]

    def lonlat_to_pixel_list(self, lonlat_list):
        """批量转换经纬度列表"""
        return [self.lonlat_to_pixel(lon, lat) for lon, lat in lonlat_list]

    def get_destination_point(self, lon, lat, dist, angle):
        """根据初始点、距离和方位角计算新点，修复 lat1 变量问题"""
        lat2 = 180 * dist * math.cos(math.radians(angle)) / (self.EARTH_RADIUS * math.pi) + lat
        lon2 = 180 * dist * math.sin(math.radians(angle)) / (self.EARTH_RADIUS * math.pi * math.cos(math.radians(lat))) + lon
        return lon2, lat2

    def get_view_corners_gps(self, lon0, lat0, height, beta, fov, aspect_ratio=(4, 3)):
        """根据航向角计算视野四个角的理论 GPS 坐标"""
        theta = math.radians(fov / 2.0)
        c = height * math.tan(theta) # 中心到角点的物理半径
        alpha = math.atan(aspect_ratio[0] / aspect_ratio[1]) * 180 / math.pi
        
        corners = []
        for i in range(4):
            angle = (-1)**(i+1) * alpha + (-180) * int((i+1)/2) + beta
            corners.append(self.get_destination_point(lon0, lat0, c, angle))
        return corners

    def get_center_info(self, corners_pixel):
        """计算视野中心的坐标：包括像素中心、投影中心(米)和经纬度"""
        avg_x = sum(p[0] for p in corners_pixel) / 4.0
        avg_y = sum(p[1] for p in corners_pixel) / 4.0
        # 计算投影坐标（即您日志中要求的 pts 12106837.3... 格式）
        px = self.transform[0] + avg_x * self.transform[1] + avg_y * self.transform[2]
        py = self.transform[3] + avg_x * self.transform[4] + avg_y * self.transform[5]
        lon, lat = self.pixel_to_lonlat(avg_x, avg_y)
        return avg_x, avg_y, lon, lat, px, py
