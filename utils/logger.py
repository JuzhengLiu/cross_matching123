# C:\Users\xiaoju\Desktop\eloft-pipei\cross_matching\utils\logger.py
# -*- coding: utf-8 -*-

import csv
import os
import cv2
import numpy as np
import math
import matplotlib.cm as cm

# 尝试从 EfficientLoFTR 库中导入官方绘图工具，用于生成匹配连线图
try:
    from src.utils.plotting import make_matching_figure
except ImportError:
    make_matching_figure = None

class MatchLogger:
    def __init__(self, config):
        """
        初始化日志记录器，支持详细的误差分析逻辑。
        """
        self.csv_path = config['paths']['output_csv']
        self.debug_dir = config['paths']['output_debug_dir']
        self.fmt = "png" 
        
        # 读取误差分析配置
        self.error_cfg = config.get('error_analysis', {})
        self.calculate_error = self.error_cfg.get('enabled', False)
        self.ground_truth = {}
        
        # WGS84 地球平均半径 (米)
        self.EARTH_RADIUS = 6371393.0

        # 如果开启误差计算，则解析真值文件
        if self.calculate_error:
            gt_path = self.error_cfg.get('ground_truth_path', "")
            if os.path.exists(gt_path):
                self._load_ground_truth(gt_path)
                print(f"[INFO] 误差分析模块已就绪，真值数据加载成功。")
            else:
                print(f"[WARN] 找不到真值文件，误差计算功能已关闭。")
                self.calculate_error = False

        # 确保输出目录完整
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # 初始化 CSV 结果文件表头
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ["image_id", "lon", "lat", "proj_x", "proj_y", "status", "cost_sec"]
                # 如果计算误差，则增加详细误差列
                if self.calculate_error:
                    header.extend(["err_lon_m", "err_lat_m", "err_combined_m"])
                writer.writerow(header)

    def _load_ground_truth(self, path):
        """内部方法：解析包含 img, lat, lng 的真值 CSV 文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # 适配 Tab 分隔符格式
                if not reader.fieldnames or len(reader.fieldnames) == 1:
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter='\t')
                
                for row in reader:
                    img_name = row['img'].strip()
                    self.ground_truth[img_name] = {
                        'lat': float(row['lat']),
                        'lng': float(row['lng'])
                    }
        except Exception as e:
            print(f"[ERROR] 真值文件解析失败: {e}")

    def _compute_detailed_errors(self, lon_pred, lat_pred, lon_gt, lat_gt):
        """
        核心数学方法：计算经度方向、纬度方向及综合物理误差（单位：米）。
        基于球面模型：1度纬度约等于 111km，1度经度约等于 111km * cos(纬度)。
        """
        # 1. 计算纬度方向物理误差 (m)
        lat_err_m = abs(lat_pred - lat_gt) * (math.pi / 180.0) * self.EARTH_RADIUS
        
        # 2. 计算经度方向物理误差 (m)
        # 注意：经度间距随纬度增加而收缩，故需乘以 cos(平均纬度)
        avg_lat = math.radians((lat_pred + lat_gt) / 2.0)
        lon_err_m = abs(lon_pred - lon_gt) * (math.pi / 180.0) * self.EARTH_RADIUS * math.cos(avg_lat)
        
        # 3. 计算综合物理误差 (m) - 使用 Haversine 公式
        dlon = math.radians(lon_gt - lon_pred)
        dlat = math.radians(lat_gt - lat_pred)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat_pred)) * math.cos(math.radians(lat_gt)) * math.sin(dlon / 2)**2
        combined_err_m = 2 * math.asin(math.sqrt(a)) * self.EARTH_RADIUS
        
        return lon_err_m, lat_err_m, combined_err_m

    def log_success(self, img_id, lon, lat, px, py, cost):
        """
        记录并输出定位成功信息，包含详细的三向误差分析。
        """
        err_data = [None, None, None]
        err_log = ""
        
        # 如果开启误差计算且真值库中有当前图片
        if self.calculate_error and img_id in self.ground_truth:
            gt = self.ground_truth[img_id]
            err_lon, err_lat, err_total = self._compute_detailed_errors(lon, lat, gt['lng'], gt['lat'])
            err_data = [f"{err_lon:.3f}", f"{err_lat:.3f}", f"{err_total:.3f}"]
            # 拼接控制台日志中的误差展示
            err_log = f" | Err: Lon={err_data[0]}m, Lat={err_data[1]}m, Total={err_data[2]}m"

        # 写入 CSV 结果文件
        with open(self.csv_path, "a", newline='', encoding='utf-8') as f:
            row = [img_id, f"{lon:.7f}", f"{lat:.7f}", f"{px:.2f}", f"{py:.2f}", 0, f"{cost:.3f}"]
            if self.calculate_error:
                row.extend(err_data)
            csv.writer(f).writerow(row)
        
        # 打印详细控制台日志，保留用户要求的 pts 格式
        print(f"[SUCCESS] {img_id} -> Lon: {lon:.6f}, Lat: {lat:.6f} | pts: {px:.1f},{py:.1f}{err_log}")

    def log_fail(self, img_id, lon, lat, px, py, cost, msg=""):
        """定位失败记录"""
        with open(self.csv_path, "a", newline='', encoding='utf-8') as f:
            row = [img_id, f"{lon:.7f}", f"{lat:.7f}", f"{px:.2f}", f"{py:.2f}", 1, f"{cost:.3f}"]
            if self.calculate_error:
                row.extend(["", "", ""])
            csv.writer(f).writerow(row)
        print(f"[FAILED] {img_id} -> 使用上一帧预测坐标 | {msg}")

    # --- save_full_overlay 和 save_matching_plot 逻辑严格保留，不删减任何功能 ---

    def save_full_overlay(self, full_base, drone_img, drone_corners_in_full, img_id, scale):
        """将无人机图像映射到全量底图大图上"""
        h, w = drone_img.shape[:2]
        src_pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
        dst_pts = np.float32(drone_corners_in_full)
        H_global = cv2.getPerspectiveTransform(src_pts, dst_pts)
        res_full = full_base.copy()
        cv2.warpPerspective(drone_img, H_global, (full_base.shape[1], full_base.shape[0]), 
                            dst=res_full, borderMode=cv2.BORDER_TRANSPARENT)
        if scale != 1.0:
            new_w, new_h = int(res_full.shape[1] * scale), int(res_full.shape[0] * scale)
            res_full = cv2.resize(res_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        save_path = os.path.join(self.debug_dir, "full_overlays", f"full_{img_id.split('.')[0]}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, res_full)
        return save_path

    def save_matching_plot(self, img0, img1, mkpts0, mkpts1, mconf, img_id):
        """保存侧并排特征连线图"""
        if make_matching_figure is None: return
        img0_rgb = cv2.cvtColor(cv2.resize(img0, (640, 480)), cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(cv2.resize(img1, (640, 480)), cv2.COLOR_BGR2RGB)
        color = cm.jet(mconf)
        text = ['EfficientLoFTR', f'Inliers: {len(mkpts0)}']
        save_path = os.path.join(self.debug_dir, "matching_plots", f"plot_{img_id.split('.')[0]}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        make_matching_figure(img0_rgb, img1_rgb, mkpts0, mkpts1, color, text=text, path=save_path)
