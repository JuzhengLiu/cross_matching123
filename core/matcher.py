# C:\Users\xiaoju\Desktop\eloft-pipei\cross_matching\core\matcher.py
# -*- coding: utf-8 -*-

import torch
import cv2
import numpy as np
import warnings
# 忽略 PyTorch 在 meshgrid 上的版本更新警告，保持控制台清洁
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# 从 EfficientLoFTR 库中导入配置和模型类
try:
    from src.loftr import LoFTR, full_default_cfg as default_cfg
except ImportError:
    raise ImportError("导入失败：请确认 environment.py 中的 eloft_root 路径指向正确的源码目录。")

class ELoFTRMatcher:
    def __init__(self, config):
        """
        初始化匹配器并加载 outdoor 预训练权重。
        """
        # 设置计算设备，优先使用 GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 从配置中读取置信度阈值和 RANSAC 容忍距离
        self.conf_thr = config['matcher']['conf_threshold']
        self.ransac_thr = config['matcher']['ransac_threshold']
        
        # 复制默认配置并强制设定精细匹配的温度系数为 10.0 以提升亚像素精度
        current_cfg = default_cfg.copy()
        current_cfg['match_fine']['local_regress_temperature'] = config['matcher'].get('fine_temperature', 10.0)
        
        # 实例化模型并加载权重文件
        self.matcher = LoFTR(config=current_cfg)
        weights_path = config['paths']['model_weights']
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # 兼容不同保存格式的权重字典
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        self.matcher.load_state_dict(state_dict, strict=False)
        self.matcher = self.matcher.eval().to(self.device)
        print(f"[INFO] ELoFTR 模型加载成功 (Device: {self.device})")

    def _preprocess(self, img, size):
        """将 BGR 图像转为灰度并归一化为 Tensor"""
        img_resized = cv2.resize(img, size)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(img_gray).float()[None, None].to(self.device) / 255.

    def match(self, img_left, img_right, size=(640, 480)):
        """
        对两张图进行特征匹配并计算单应性矩阵。
        :return: 角点像素坐标, 单应性矩阵, 匹配对数, 纠偏预览图, 原始点对数据
        """
        # 预处理两张图片
        inp0 = self._preprocess(img_left, size)
        inp1 = self._preprocess(img_right, size)
        
        batch = {'image0': inp0, 'image1': inp1}
        with torch.no_grad():
            self.matcher(batch)
            if 'mkpts0_f' not in batch:
                return None, None, 0, None, None
            
            # 提取模型输出的特征点坐标和置信度
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

        # 根据配置的阈值筛选高质量匹配点
        valid = mconf > self.conf_thr
        mkpts0, mkpts1, mconf = mkpts0[valid], mkpts1[valid], mconf[valid]

        # 如果匹配点数少于 RANSAC 计算所需的最小值，则判定失败
        if len(mkpts0) < 10:
            return None, None, len(mkpts0), None, None

        # 计算单应性矩阵 H，获取内点掩码 mask 用于后期绘图
        H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, self.ransac_thr)
        if H is None:
            return None, None, len(mkpts0), None, None

        # 提取 RANSAC 筛选后的内点数据，供绘图使用
        mask = mask.flatten()
        inliers_idx = np.where(mask == 1)[0]
        match_data = (mkpts0[inliers_idx], mkpts1[inliers_idx], mconf[inliers_idx])

        # 在卫星裁剪块上透明叠加无人机变形图，解决底图变黑问题
        res_vis = cv2.resize(img_right, size)
        cv2.warpPerspective(img_left, H, size, dst=res_vis, borderMode=cv2.BORDER_TRANSPARENT)
        
        # 计算无人机图四个角在基准图中的位置
        h, w = size[1], size[0]
        src_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst_pts = cv2.perspectiveTransform(src_pts, H).reshape(4, 2).tolist()
        
        return dst_pts, H, len(mkpts0), res_vis, match_data