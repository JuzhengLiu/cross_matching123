# C:\Users\xiaoju\Desktop\eloft-pipei\cross_matching\main.py
# -*- coding: utf-8 -*-

import os
import cv2
import time
import csv
import numpy as np

# 1. 加载运行环境
from environment import setup_environment
config = setup_environment()

# 2. 导入核心模块
from core.matcher import ELoFTRMatcher
from core.geometry import GeoTransformer
from core.processor import ImageProcessor
from utils.logger import MatchLogger

def load_sensor_data(csv_path):
    """
    读取传感器数据文件，返回 {filename: yaw_angle} 字典。
    兼容逗号分隔(CSV)或制表符分隔。
    """
    sensor_map = {}
    if not os.path.exists(csv_path):
        print(f"[WARN] 传感器文件未找到: {csv_path}，将回退到纯视觉预测模式。")
        return sensor_map
        
    print(f"[INFO] 正在加载传感器数据: {csv_path}")
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig 自动处理 BOM
            # 尝试自动检测分隔符，防止 Excel 保存格式不同
            line = f.readline()
            dialect = csv.Sniffer().sniff(line) if line else 'excel'
            f.seek(0)
            
            reader = csv.DictReader(f, dialect=dialect)
            # 标准化列名：移除空格，转小写
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            
            for row in reader:
                # 兼容不同的列名写法
                fname = row.get('FileName', row.get('filename', row.get('img', ''))).strip()
                # 优先找 'yaw'，其次找 'yaw(deg)'，最后根据您提供的数据找 'yaw'
                yaw_val = row.get('yaw', row.get('Yaw', row.get('机身偏航', None)))
                
                if fname and yaw_val is not None:
                    try:
                        sensor_map[fname] = float(yaw_val)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[ERROR] 读取传感器 CSV 失败: {e}")
    
    print(f"[INFO] 已加载 {len(sensor_map)} 帧传感器姿态数据。")
    return sensor_map

def main():
    """
    定位系统主入口：支持传感器辅助的剧烈机动定位。
    """
    # 初始化模块
    matcher = ELoFTRMatcher(config)
    geo_tool = GeoTransformer(config['paths']['base_image'])
    processor = ImageProcessor()
    logger = MatchLogger(config)
    
    # 加载基准图
    full_base = cv2.imread(config['paths']['base_image'])
    if full_base is None:
        print("[CRITICAL] 无法加载基准图，请核对 yml 路径。")
        return

    # 加载传感器偏航角数据
    sensor_csv_path = config['paths'].get('sensor_data', '')
    yaw_data_map = load_sensor_data(sensor_csv_path)

    # 【初始阶段】
    init = config['initial_state']
    # 优先从传感器数据中获取第一帧的 yaw，如果没有则用配置文件的 beta_o
    start_yaw = init['beta_o']
    
    # 计算第一帧的理论视野
    curr_height = init['h_o'] - init['h0']
    corners_gps = geo_tool.get_view_corners_gps(
        init['lng_o'], init['lat_o'], curr_height, start_yaw, config['camera']['fov']
    )
    img_xy = geo_tool.lonlat_to_pixel_list(corners_gps)
    
    # 记录上一帧的地理坐标，用于推算下一帧位置
    prev_lon, prev_lat = init['lng_o'], init['lat_o']
    prev_xy = img_xy # 像素坐标备份

    # 扫描图像列表
    input_dir = config['paths']['realtime_dir']
    img_list = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    
    print(f"\n[SYSTEM] 启动处理 | 传感器辅助模式: {'ON' if yaw_data_map else 'OFF'}")

    for img_name in img_list:
        t_start = time.time()
        frame = cv2.imread(os.path.join(input_dir, img_name))
        if frame is None: continue

        # --- 0. 图像预处理 (旋转与缩放) ---
        # 根据配置文件决定是否旋转输入图 (解决 Roll=0 vs Roll=180 问题)
        req_rotation = config['camera'].get('image_rotation', 0)
        if req_rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # 统一缩放到配置分辨率
        target_res = tuple(config['camera']['resolution'])
        if frame.shape[:2][::-1] != target_res:
            frame = cv2.resize(frame, target_res)

        # --- 1. 动态搜索区域计算 (核心修改) ---
        # 确定当前帧的偏航角
        current_yaw = start_yaw # 默认值
        if img_name in yaw_data_map:
            current_yaw = yaw_data_map[img_name]
            # print(f"[{img_name}] Sensor Yaw: {current_yaw}")
        
        # 只要不是第一帧，就利用“上一帧位置 + 当前传感器角度”重新计算搜索框
        # 这能确保即使飞机转了90度，裁剪出来的卫星图也是转了90度的
        if 'prev_lon' in locals():
            # 使用 geo_tool 重新计算具备正确旋转角度的 GPS 视野四角
            corners_gps = geo_tool.get_view_corners_gps(
                prev_lon, prev_lat, curr_height, current_yaw, config['camera']['fov']
            )
            # 转为像素坐标
            tight_xy = geo_tool.lonlat_to_pixel_list(corners_gps)
            # 稍微扩大一点范围 (mag_k)，防止 GPS 误差导致出框
            img_xy = processor.expand_rect(tight_xy, config['matcher'].get('mag_k', 2.0))
        
        # --- 2. 基准图裁剪与对齐 ---
        base_roi, roi_y, roi_x = processor.get_base_roi(full_base, img_xy)
        local_corners = [[p[0]-roi_x, p[1]-roi_y] for p in img_xy]
        
        # 这里 rotate_and_crop 会根据 img_xy (已经包含了 current_yaw 信息) 自动计算旋转
        base_crop, rot_m, crop_y, crop_x = processor.rotate_and_crop(base_roi, local_corners)
        
        # --- 3. EfficientLoFTR 匹配 ---
        dst_pts_crop, G, count, _, match_data = matcher.match(frame, base_crop)
        
        # --- 4. 结果处理 ---
        is_valid = processor.validate_homography(G, config['matcher']['valid_k'])
        
        if is_valid and dst_pts_crop is not None:
            # 坐标逆向还原
            actual_xy = processor.map_points_back(dst_pts_crop, base_crop.shape, base_roi.shape, rot_m, crop_y, crop_x, roi_y, roi_x)
            
            # 获取地理坐标
            _, _, lon, lat, px, py = geo_tool.get_center_info(actual_xy)
            
            # 【关键】更新上一帧位置，供下一帧结合新 Yaw 使用
            prev_lon, prev_lat = lon, lat
            prev_xy = actual_xy 
            
            logger.log_success(img_name, lon, lat, px, py, time.time() - t_start)
            
            if config['matcher'].get('save_full_overlay', True):
                logger.save_full_overlay(full_base, frame, actual_xy, img_name, config['matcher'].get('full_overlay_scale', 0.5))

            if config['matcher'].get('save_matching_fig', True) and match_data:
                logger.save_matching_plot(frame, base_crop, *match_data, img_name)
        else:
            # 匹配失败：沿用上一帧位置，但下一帧循环时会用新 Yaw 再次尝试
            _, _, lon, lat, px, py = geo_tool.get_center_info(prev_xy)
            logger.log_fail(img_name, lon, lat, px, py, time.time() - t_start, f"Pts:{count}")

    print(f"\n[SYSTEM] 任务全部处理完成。")

if __name__ == "__main__":
    main()