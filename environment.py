# C:\Users\xiaoju\Desktop\eloft-pipei\cross_matching\environment.py
# -*- coding: utf-8 -*-

import sys
import os
import yaml
from pathlib import Path

def setup_environment(config_path="config/default_config_input3.yml"):
#def setup_environment(config_path="config/default_config.yml"):
    """
    配置运行环境：
    1. 加载 YAML 配置文件。
    2. 设置 GDAL/PROJ 环境变量，确保坐标转换词典 proj.db 可用。
    3. 动态添加 EfficientLoFTR 代码库到 Python 路径。
    """
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件，请检查路径: {config_path}")
    
    # 读取 YAML 配置文件内容
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取当前虚拟环境的基础路径，用于定位 GDAL 数据集
    # 这里使用您之前找到的 site-packages 下的确切路径
    env_base = r"D:\7_deepl\environment\eloftr"
    proj_path = os.path.join(env_base, "Lib", "site-packages", "osgeo", "data", "proj")
    gdal_path = os.path.join(env_base, "Lib", "site-packages", "osgeo", "data", "gdal")
    
    # 强制设置环境变量，防止 GDAL 在执行坐标转换时报错 Cannot find proj.db
    if os.path.exists(proj_path):
        os.environ['PROJ_LIB'] = proj_path
        os.environ['GDAL_DATA'] = gdal_path
        print(f"[INFO] 环境变量已锁定: {proj_path}")
    
    # 将外部的 EfficientLoFTR 源码库加入系统路径，以便直接 import src
    eloft_path = config['paths']['eloft_root']
    if eloft_path not in sys.path:
        sys.path.append(eloft_path)
        print(f"[INFO] 已挂载高效匹配库: {eloft_path}")
    
    return config