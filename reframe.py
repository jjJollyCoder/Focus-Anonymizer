import cv2
import numpy as np
import torch
from config import *
from utils import MovingAverageFilter

class ReframeTracker:
    def __init__(self):
        # 1. 自动选择最佳算力后端 (仅在 Apple Silicon M 系列芯片上启用 MPS，Intel Mac 强制采用 CPU)
        import platform
        is_intel = platform.machine() == "x86_64"
        if torch.backends.mps.is_available() and not is_intel:
            self.device = "mps"
            print("🧭 军师检测：发现 Apple Silicon GPU，启用 Metal (MPS) 硬件加速检测！")
        elif torch.cuda.is_available():
            self.device = "0"
            print("🧭 军师检测：发现 NVIDIA GPU，启用 CUDA 硬件加速检测！")
        else:
            self.device = "cpu"
            if is_intel:
                print("🧭 军师检测：检测到当前为 Intel 架构 Mac，为了绝对稳定并规避 AMD 显卡驱动死锁，强制采用 CPU 向量指令集 (AVX2) 运行检测！")
            else:
                print("🧭 军师检测：未检测到专属 GPU，采用 Mac CPU 跑模型。")
            
        # 延迟导入，防止启动加载时间过长
        from ultralytics import YOLO
        self.model = YOLO(YOLO_MODEL)
        
        # 记录上一帧的参数，用于惯性缓动平滑 (Inertia Easing / Lerp)
        self.prev_center_x = None
        self.prev_center_y = None
        self.prev_crop_w = None
        self.prev_crop_h = None
        
        # 记录最近一次定位到的主体边界框，用于连续帧的最近邻追踪
        self.last_bbox = None
        
        # 记录当前帧检测到的所有人体 Bounding Box，用于一键联动模糊挡脸，规避双重 YOLO 加载
        self.all_candidates = []
        
        # 记录是否完成了首帧的主体锁定（Scheme 2：交互指指标签）
        self.is_initialized = False

    def reset_tracking(self, preserve_last_bbox=True):
        """场景切换或重置时清空跟踪历史，防止跨场景过渡发生瞬间跳变"""
        self.prev_center_x = None
        self.prev_center_y = None
        self.prev_crop_w = None
        self.prev_crop_h = None
        if not preserve_last_bbox:
            self.last_bbox = None
            self.is_initialized = False

    def detect_and_track_person(self, frame):
        """
        在帧中定位并追踪目标人物（你）。返回目标人物的 Bounding Box (x1, y1, x2, y2)
        """
        results = self.model(frame, verbose=False, device=self.device)
        boxes = results[0].boxes
        
        candidates = []
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            if cls_id == PERSON_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
                xyxy = box.xyxy[0].cpu().numpy()
                candidates.append(xyxy)
                
        # 记录当前帧检出的所有候选人体，共享给 mask 模块，杜绝双重运行 YOLO
        self.all_candidates = candidates

        if not candidates:
            # 如果这一帧没检测到任何人，将候选置空，并返回最近一次的历史定位框作为兜底
            self.all_candidates = []
            return self.last_bbox

        # 如果有多个候选人，且是第一次处理（is_initialized 为 False），启动交互指认（Scheme 2）
        if not self.is_initialized:
            if len(candidates) == 1:
                # 只有一个人，无需指认，直接自动选
                self.last_bbox = candidates[0]
                self.is_initialized = True
                print("🧭 军师检测：画面中只有 1 个人物，已自动锁定该主体！")
            else:
                # 画面有多个候选人，进行交互式指认！
                print(f"\n🧭 军师检测到画面中共有 {len(candidates)} 个人物。")
                
                # 绘制有编号的候选图
                annotated_frame = frame.copy()
                for idx, xy in enumerate(candidates):
                    x1, y1, x2, y2 = map(int, xy)
                    # 绘制边界框
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # 绘制红色编号标签
                    label = f"[{idx}]"
                    cv2.putText(annotated_frame, label, (x1 + 10, y1 + 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    
                    # 打印控制台说明：说明每个候选人在画面中的坐标和方位
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    width_ratio = (cx / frame.shape[1]) * 100
                    print(f"   👤 候选人 [{idx}]: 位于画面横向 {width_ratio:.1f}% 处 (x={int(cx)}, y2={y2})")

                # 将首帧指认效果图自动保存到项目输出文件夹中
                output_dir = "/Users/jojo/Downloads/short_video_automation/output_videos"
                os.makedirs(output_dir, exist_ok=True)
                preview_path = os.path.join(output_dir, "first_frame_select.png")
                cv2.imwrite(preview_path, annotated_frame)
                
                # 保姆级体验：自动利用 macOS 系统命令打开该预览图，并在前台展现给用户！
                import subprocess
                try:
                    subprocess.run(["open", preview_path])
                    print(f"🚀 军师提醒：已为您在屏幕上自动打开候选人位置预览图 [first_frame_select.png]！")
                    print(f"   (如果未看到，可手动前往该路径双击打开: {preview_path})")
                except Exception:
                    pass
                
                # 在控制台进行阻塞式输入指认
                while True:
                    try:
                        user_input = input(f"\n👉 请输入您本人的编号 [0-{len(candidates)-1}] 进行追踪锁定: ")
                        select_idx = int(user_input.strip())
                        if 0 <= select_idx < len(candidates):
                            self.last_bbox = candidates[select_idx]
                            self.is_initialized = True
                            print(f"🎯 锁定成功！已将候选人 [{select_idx}] 绑定为您的追踪主体。")
                            break
                        else:
                            print(f"❌ 编号不正确，请输入 0 到 {len(candidates)-1} 之间的整数。")
                    except ValueError:
                        print("❌ 输入格式有误，请输入纯数字编号。")
        else:
            # 已经完成了初始化：后续帧或场景切换（reset_tracking 但保持 last_bbox）
            if self.last_bbox is None:
                # 场景切换极端兜底：如果没有保留 last_bbox，采用默认最中央人物
                frame_center_x = frame.shape[1] / 2
                candidates_center_x = [(xy[0] + xy[2]) / 2 for xy in candidates]
                best_idx = np.argmin([abs(cx - frame_center_x) for cx in candidates_center_x])
                self.last_bbox = candidates[best_idx]
            else:
                # 后续帧：寻找与上一帧 BBox 距离最近的人进行锁定追踪
                prev_center = np.array([(self.last_bbox[0] + self.last_bbox[2])/2, (self.last_bbox[1] + self.last_bbox[3])/2])
                distances = []
                for xy in candidates:
                    curr_center = np.array([(xy[0] + xy[2])/2, (xy[1] + xy[3])/2])
                    dist = np.linalg.norm(curr_center - prev_center)
                    distances.append(dist)
                best_idx = np.argmin(distances)
                self.last_bbox = candidates[best_idx]
            
        return self.last_bbox

    def calculate_crop_box(self, frame, mode="pan_only"):
        """
        根据追踪到的人物计算 9:16 的裁剪区域。
        mode = "pan_only": 仅做横向平移，高度保持100%原图（最稳定、画质最高）。
        mode = "pan_zoom": 平移+拉伸缩放，始终保持人物在画面中的垂直占比恒定（复刻剪映“人物居中锁定”）。
        """
        img_h, img_w, _ = frame.shape
        bbox = self.detect_and_track_person(frame)
        
        # 默认 9:16 基础分辨率比率
        aspect_ratio = 9 / 16
        
        if bbox is None:
            # 极限兜底：如果没有检测到人，默认对齐物理中心
            center_x = img_w / 2
            center_y = img_h / 2
            crop_h = img_h
            crop_w = int(img_h * aspect_ratio)
        else:
            bx1, by1, bx2, by2 = bbox
            person_cx = (bx1 + bx2) / 2
            person_cy = (by1 + by2) / 2
            person_h = by2 - by1
            
            if mode == "pan_only":
                # 仅平移：高度为视频全高，宽度按照 9:16 换算
                crop_h = img_h
                crop_w = int(img_h * aspect_ratio)
                center_x = person_cx
                center_y = img_h / 2
            else: # pan_zoom
                # 居中锁定：假设让人物高度始终在画面中占比 55%
                target_person_ratio = 0.55
                crop_h = int(person_h / target_person_ratio)
                # 限制裁剪框不能超过视频本身的边界
                crop_h = min(img_h, max(int(img_h * 0.4), crop_h)) 
                crop_w = int(crop_h * aspect_ratio)
                
                center_x = person_cx
                center_y = person_cy

        # 2. 惯性缓动平滑 (Lerp - Linear Interpolation)
        # 如果是第一帧，直接初始化镜头坐标
        if self.prev_center_x is None:
            self.prev_center_x = center_x
            self.prev_center_y = center_y
            self.prev_crop_w = crop_w
            self.prev_crop_h = crop_h
        
        # 缓动系数 (Easing factor): 越小越缓和/平滑，越大越紧跟。0.06 是电影感慢速平滑推移镜头的黄金系数
        easing_factor = 0.06
        
        scx = self.prev_center_x + easing_factor * (center_x - self.prev_center_x)
        scy = self.prev_center_y + easing_factor * (center_y - self.prev_center_y)
        scw = int(self.prev_crop_w + easing_factor * (crop_w - self.prev_crop_w))
        sch = int(self.prev_crop_h + easing_factor * (crop_h - self.prev_crop_h))

        self.prev_center_x = scx
        self.prev_center_y = scy
        self.prev_crop_w = scw
        self.prev_crop_h = sch

        # 3. 边界处理，确保裁剪区域不超出原始图像物理宽度和高度
        # 计算左上角起点 (x1, y1)
        cx1 = int(scx - scw / 2)
        cy1 = int(scy - sch / 2)
        
        # 边界校正与对齐
        if cx1 < 0:
            cx1 = 0
        if cy1 < 0:
            cy1 = 0
        if cx1 + scw > img_w:
            cx1 = img_w - scw
        if cy1 + sch > img_h:
            cy1 = img_h - sch

        # 再次确保长宽为偶数（FFmpeg 及部分视频播放器对奇数长宽可能不兼容）
        scw = (scw // 2) * 2
        sch = (sch // 2) * 2
        
        return cx1, cy1, scw, sch
