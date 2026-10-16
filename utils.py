import cv2
import numpy as np

class MovingAverageFilter:
    """
    滑动平均滤波器，用于对检测到的坐标、宽度等参数进行平滑，消除画面抖动。
    """
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.history = []
        
    def update(self, val):
        self.history.append(val)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        # 支持一维数值或元组/列表等多维数值的平滑
        return np.mean(self.history, axis=0)


def overlay_sticker(bg_img, sticker, center_x, center_y, target_w, angle=0):
    """
    将透明通道 PNG 贴纸完美缩放、旋转并叠加到背景图帧上（支持越界裁剪）。
    """
    h, w, c = sticker.shape
    if c < 4:
        # 如果贴纸没有透明通道，转为 BGRA
        sticker = cv2.cvtColor(sticker, cv2.COLOR_BGR2BGRA)
    
    # 保持宽高比进行缩放
    aspect_ratio = h / w
    target_h = int(target_w * aspect_ratio)
    if target_w <= 0 or target_h <= 0:
        return bg_img
    
    sticker_resized = cv2.resize(sticker, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # 带有旋转
    if angle != 0:
        # 旋转可能使贴纸四角超出边界，构建更大的正方形内边距容器
        diagonal = int(np.sqrt(target_w**2 + target_h**2))
        pad_x = (diagonal - target_w) // 2
        pad_y = (diagonal - target_h) // 2
        padded = cv2.copyMakeBorder(sticker_resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0,0,0,0])
        
        center = (padded.shape[1] // 2, padded.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        sticker_rotated = cv2.warpAffine(padded, rot_mat, (padded.shape[1], padded.shape[0]), 
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])
    else:
        sticker_rotated = sticker_resized

    # 贴纸的长宽与中点定位
    sr_h, sr_w, _ = sticker_rotated.shape
    start_x = int(center_x - sr_w / 2)
    start_y = int(center_y - sr_h / 2)
    
    bg_h, bg_w, _ = bg_img.shape
    
    # 边界裁剪：计算贴纸和背景图的交集区域，避免坐标越界时程序奔溃
    sticker_x1 = max(0, -start_x)
    sticker_y1 = max(0, -start_y)
    sticker_x2 = min(sr_w, bg_w - start_x)
    sticker_y2 = min(sr_h, bg_h - start_y)
    
    bg_x1 = max(0, start_x)
    bg_y1 = max(0, start_y)
    bg_x2 = min(bg_w, start_x + sr_w)
    bg_y2 = min(bg_h, start_y + sr_h)
    
    if (sticker_x2 <= sticker_x1) or (sticker_y2 <= sticker_y1):
        return bg_img  # 完全在画面外，直接返回原图
        
    sticker_crop = sticker_rotated[sticker_y1:sticker_y2, sticker_x1:sticker_x2]
    bg_crop = bg_img[bg_y1:bg_y2, bg_x1:bg_x2]
    
    # 提取 Alpha 透明通道
    alpha_sticker = sticker_crop[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_sticker
    
    # 像素融合
    for channel in range(3):
        bg_crop[:, :, channel] = (alpha_sticker * sticker_crop[:, :, channel] + 
                                  alpha_bg * bg_crop[:, :, channel]).astype(np.uint8)
        
    bg_img[bg_y1:bg_y2, bg_x1:bg_x2] = bg_crop
    return bg_img
