import cv2
import numpy as np
from config import *

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0
    return interArea / unionArea

class FaceMasker:
    def __init__(self):
        print("🧭 军师提醒：坐标与 IoU 双重安全锁定遮罩加载成功！")

    def apply_face_masks(self, frame, all_candidates, my_bbox, crop_box):
        if not all_candidates:
            return frame

        img_h, img_w, _ = frame.shape
        crop_x, crop_y, crop_w, crop_h = crop_box
        scale_x = img_w / crop_w
        scale_y = img_h / crop_h

        for bbox in all_candidates:
            bx1, by1, bx2, by2 = bbox
            body_w = bx2 - bx1
            body_h = by2 - by1
            
            if body_w <= 0 or body_h <= 0:
                continue
                
            # 🚀 【极速高精度过滤我本人】：
            if my_bbox is not None:
                # 1. 坐标严格重合（代表这是同一帧中 YOLO 分配给我的候选框）
                if np.allclose(bbox, my_bbox, atol=2.0):
                    continue
                # 2. 或者交并比重合度极高（安全冗余）
                if calculate_iou(bbox, my_bbox) > 0.8:
                    continue

            # 视口空间坐标映射
            projected_bx1 = (bx1 - crop_x) * scale_x
            projected_by1 = (by1 - crop_y) * scale_y
            projected_bx2 = (bx2 - crop_x) * scale_x
            projected_by2 = (by2 - crop_y) * scale_y
            
            p_body_w = projected_bx2 - projected_bx1
            p_body_h = projected_by2 - projected_by1
            p_cx = (projected_bx1 + projected_bx2) / 2

            # 视口裁剪过滤 (Frustum Culling)
            if (projected_bx2 < -10) or (projected_bx1 > img_w + 10) or \
               (projected_by2 < -10) or (projected_by1 > img_h + 10):
                continue

            # 计算 1080P 下头部的精确物理椭圆几何参数
            head_cx = int(p_cx)
            head_cy = int(projected_by1 + p_body_h * 0.08)
            head_rx = int(p_body_w * 0.35)
            head_ry = int(p_body_h * 0.12)
            
            if head_rx <= 2 or head_ry <= 2:
                continue

            # 局部 ROI 极速模糊
            rx1 = max(0, head_cx - head_rx)
            ry1 = max(0, head_cy - head_ry)
            rx2 = min(img_w, head_cx + head_rx)
            ry2 = min(img_h, head_cy + head_ry)
            
            roi_w = rx2 - rx1
            roi_h = ry2 - ry1
            if roi_w <= 0 or roi_h <= 0:
                continue
                
            roi = frame[ry1:ry2, rx1:rx2].copy()
            blur_kernel_size = int(head_rx * 1.5) | 1
            if blur_kernel_size < 5:
                blur_kernel_size = 5
            blurred_roi = cv2.GaussianBlur(roi, (blur_kernel_size, blur_kernel_size), 0)
            
            mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
            local_cx = head_cx - rx1
            local_cy = head_cy - ry1
            cv2.ellipse(mask_roi, (local_cx, local_cy), (head_rx, head_ry), 0, 0, 360, 255, -1)
            
            feather_size = int(head_rx * 0.4) | 1
            if feather_size < 3:
                feather_size = 3
            mask_roi_feathered = cv2.GaussianBlur(mask_roi, (feather_size, feather_size), 0)
            
            alpha = mask_roi_feathered.astype(float) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            blended_roi = (alpha * blurred_roi + (1.0 - alpha) * roi).astype(np.uint8)
            frame[ry1:ry2, rx1:rx2] = blended_roi
            
        return frame
