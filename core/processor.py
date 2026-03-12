# C:\Users\xiaoju\Desktop\eloft-pipei\cross_matching\core\processor.py
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

class ImageProcessor:
    @staticmethod
    def get_base_roi(full_base_img, corners_xy):
        """从全图中提取初始搜索区域 (ROI)"""
        h_list = [p[1] for p in corners_xy]
        w_list = [p[0] for p in corners_xy]
        y_start, y_end = int(min(h_list)), int(max(h_list))
        x_start, x_end = int(min(w_list)), int(max(w_list))
        H_max, W_max = full_base_img.shape[:2]
        roi = full_base_img[max(0, y_start):min(H_max, y_end), max(0, x_start):min(W_max, x_end)]
        return roi, max(0, y_start), max(0, x_start)

    @staticmethod
    def rotate_and_crop(img, corners_xy):
        """执行航向角旋转对齐并裁剪核心匹配块"""
        pt1, pt4 = corners_xy[0], corners_xy[3]
        width_rect = math.sqrt((pt4[0] - pt1[0])**2 + (pt4[1] - pt1[1])**2)
        angle = math.acos((pt4[0] - pt1[0]) / (width_rect + 1e-6)) * (180 / math.pi)
        if pt4[1] <= pt1[1]: angle = -angle
        
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        rotate_mat = cv2.getRotationMatrix2D(center, angle, 1)
        
        # 补偿画布旋转后的尺寸变化
        cos_a, sin_a = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
        new_w, new_h = int(h*sin_a + w*cos_a), int(w*sin_a + h*cos_a)
        rotate_mat[0, 2] += (new_w - w) / 2
        rotate_mat[1, 2] += (new_h - h) / 2
        
        img_rotated = cv2.warpAffine(img, rotate_mat, (new_w, new_h), borderValue=(255, 255, 255))
        
        # 转换角点坐标，确定裁剪区域
        pts_ones = np.hstack([np.array(corners_xy), np.ones((4, 1))])
        pts_rot = pts_ones.dot(rotate_mat.T)
        x_min, y_min = np.min(pts_rot, axis=0)
        x_max, y_max = np.max(pts_rot, axis=0)
        
        crop_y, crop_x = int(y_min), int(x_min)
        img_cropped = img_rotated[max(0, crop_y):int(y_max), max(0, crop_x):int(x_max)]
        return img_cropped, rotate_mat, max(0, crop_y), max(0, crop_x)

    @staticmethod
    def map_points_back(pts_in_crop, crop_size, roi_size, rotate_mat, crop_off_y, crop_off_x, roi_off_y, roi_off_x):
        """
        [最核心逻辑] 逆向映射：将 640x480 的匹配点坐标还原回全量基准图 (BaseImage)。
        链条：模型点 -> 还原缩放 -> 补偿裁剪偏置 -> 逆旋转 -> 补偿全图 ROI 偏置
        """
        if pts_in_crop is None: return None
        h_crop_img, w_crop_img = crop_size[:2]
        
        # 1. 还原 640x480 的比例缩放
        pts_scaled = [[p[0]*w_crop_img/640.0, p[1]*h_crop_img/480.0] for p in pts_in_crop]
        # 2. 补偿旋转裁剪偏置
        pts_in_rot = [[p[0]+crop_off_x, p[1]+crop_off_y] for p in pts_scaled]
        # 3. 执行逆旋转 (逆矩阵变换)
        inv_rot_mat = cv2.invertAffineTransform(rotate_mat)
        pts_in_roi = []
        for p in pts_in_rot:
            p_roi = inv_rot_mat.dot(np.array([p[0], p[1], 1.0]))
            pts_in_roi.append(p_roi.tolist())
        # 4. 补偿 ROI 在大图中的位置
        return [[p[0]+roi_off_x, p[1]+roi_off_y] for p in pts_in_roi]

    # @staticmethod
    # def validate_homography(H, valid_range=(0.5, 1.5)):
    #     """校验单应性矩阵，防止出现病态拉伸"""
    #     if H is None: return False
    #     scale_x, scale_y = abs(H[0,0]), abs(H[1,1])
    #     return valid_range[0] < scale_x < valid_range[1] and valid_range[0] < scale_y < valid_range[1]
    @staticmethod
    def validate_homography(H, valid_range=(0.2, 2.5)): # 建议范围
        """校验单应性矩阵（改进版）：使用行列式计算尺度，抗旋转干扰"""
        if H is None: return False
        # 计算旋转无关的真实尺度因子
        det = np.linalg.det(H[:2, :2])
        scale = np.sqrt(abs(det))
        
        # 只要整体缩放比例在合理范围内即可
        return valid_range[0] < scale < valid_range[1]

    @staticmethod
    def expand_rect(corners_xy, mag_k=1.1):
        """每一帧稍微扩大搜索范围，提供更好的预测鲁棒性"""
        cx, cy = sum(p[0] for p in corners_xy)/4.0, sum(p[1] for p in corners_xy)/4.0
        return [[cx + (x - cx) * mag_k, cy + (y - cy) * mag_k] for x, y in corners_xy]