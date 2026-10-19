import cv2
import numpy as np
import torch
from config import *
from utils import MovingAverageFilter

def calculate_iou(boxA, boxB):
    """计算两个边界框的 Intersection over Union (IoU) 交并比"""
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

class ReframeTracker:
    def __init__(self, select_index=None):
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
            
        from ultralytics import YOLO
        self.model = YOLO(YOLO_MODEL)
        
        self.prev_center_x = None
        self.prev_center_y = None
        self.prev_crop_w = None
        self.prev_crop_h = None
        
        self.last_bbox = None
        self.all_candidates = []
        self.is_initialized = False
        self.select_index = select_index
        
        # 🚀 【神级性能提升：YOLO 隔帧检测参数】
        self.frame_counter = 0
        self.detect_interval = 2  # 🧭 每 2 帧跑一次 YOLO 检测，CPU 负荷瞬间暴降 50%！

    def reset_tracking(self, preserve_last_bbox=True):
        self.prev_center_x = None
        self.prev_center_y = None
        self.prev_crop_w = None
        self.prev_crop_h = None
        if not preserve_last_bbox:
            self.last_bbox = None
            self.is_initialized = False
            self.frame_counter = 0

    def detect_and_track_person(self, frame):
        """利用 YOLO 隔帧自适应检测 + 多维打分 + 惰性维持，榨干 CPU 最后一滴极限算力"""
        self.frame_counter += 1
        
        # 🚀 隔帧快速通道：如果是跳过的帧，且已经完成初始化，直接复用上一帧结果，100% 豁免 YOLO 推理！
        if self.is_initialized and (self.frame_counter % self.detect_interval != 0):
            return self.last_bbox

        img_h, img_w, _ = frame.shape
        
        # 降采样特征提取：960x540 运算像素骤降 16 倍，直接让 CPU 检测速度提升 4-5 倍！
        detect_w, detect_h = 960, 540
        scale_x = img_w / detect_w
        scale_y = img_h / detect_h
        small_frame = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR)
        
        results = self.model(small_frame, verbose=False, device=self.device)
        boxes = results[0].boxes
        
        candidates = []
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            if cls_id == PERSON_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
                xyxy = box.xyxy[0].cpu().numpy()
                xyxy_scaled = np.array([
                    xyxy[0] * scale_x,
                    xyxy[1] * scale_y,
                    xyxy[2] * scale_x,
                    xyxy[3] * scale_y
                ])
                candidates.append(xyxy_scaled)
                
        self.all_candidates = candidates

        if not candidates:
            self.all_candidates = []
            return self.last_bbox

        if not self.is_initialized:
            if len(candidates) == 1:
                self.last_bbox = candidates[0]
                self.is_initialized = True
                print("🧭 军师检测：画面中只有 1 个人物，已自动锁定该主体！")
            elif self.select_index is not None:
                sel_idx = min(len(candidates) - 1, max(0, self.select_index))
                self.last_bbox = candidates[sel_idx]
                self.is_initialized = True
                print(f"🎯 [Headless Mode] 已直接将候选人 [{sel_idx}] 绑定为您的追踪主体！")
            else:
                print(f"\n🧭 军师检测到画面中共有 {len(candidates)} 个人物。")
                annotated_frame = frame.copy()
                for idx, xy in enumerate(candidates):
                    x1, y1, x2, y2 = map(int, xy)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"[{idx}]"
                    cv2.putText(annotated_frame, label, (x1 + 10, y1 + 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    
                    cx = (x1 + x2) / 2
                    width_ratio = (cx / frame.shape[1]) * 100
                    print(f"   👤 候选人 [{idx}]: 位于画面横向 {width_ratio:.1f}% 处 (x={int(cx)}, y2={y2})")

                output_dir = "/Users/jojo/Downloads/short_video_automation/output_videos"
                os.makedirs(output_dir, exist_ok=True)
                preview_path = os.path.join(output_dir, "first_frame_select.png")
                cv2.imwrite(preview_path, annotated_frame)
                
                import subprocess
                try:
                    subprocess.run(["open", preview_path])
                    print(f"🚀 军师提醒：已自动打开位置预览图 [first_frame_select.png]！")
                except:
                    pass
                
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
                            print(f"❌ 编号不正确。")
                    except ValueError:
                        print("❌ 输入格式有误。")
        else:
            # 🚀 【多维加权打分 + 物理门控 + 惰性维持算法】
            if self.last_bbox is None:
                frame_center_x = frame.shape[1] / 2
                candidates_center_x = [(xy[0] + xy[2]) / 2 for xy in candidates]
                best_idx = np.argmin([abs(cx - frame_center_x) for cx in candidates_center_x])
                self.last_bbox = candidates[best_idx]
            else:
                prev_w = self.last_bbox[2] - self.last_bbox[0]
                prev_h = self.last_bbox[3] - self.last_bbox[1]
                prev_center = np.array([(self.last_bbox[0] + self.last_bbox[2])/2, (self.last_bbox[1] + self.last_bbox[3])/2])
                
                best_score = -1e9
                best_candidate = None
                
                max_gate_distance = prev_h * 0.45
                
                for xy in candidates:
                    curr_w = xy[2] - xy[0]
                    curr_h = xy[3] - xy[1]
                    curr_center = np.array([(xy[0] + xy[2])/2, (xy[1] + xy[3])/2])
                    
                    iou = calculate_iou(xy, self.last_bbox)
                    dist = np.linalg.norm(curr_center - prev_center)
                    
                    if dist > max_gate_distance and iou < 0.15:
                        continue
                        
                    norm_dist = dist / max(1.0, prev_h)
                    size_ratio_w = min(curr_w, prev_w) / max(1.0, curr_w, prev_w)
                    size_ratio_h = min(curr_h, prev_h) / max(1.0, curr_h, prev_h)
                    size_similarity = (size_ratio_w + size_ratio_h) / 2.0
                    
                    score = iou * 6.0 + size_similarity * 2.5 - norm_dist * 2.5
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = xy
                
                if best_candidate is not None:
                    self.last_bbox = best_candidate
                    self.lost_frames_count = 0
                else:
                    self.lost_frames_count = getattr(self, "lost_frames_count", 0) + 1
                    if self.lost_frames_count > 15:  # 连续 15 帧跟丢重新初始化
                        self.is_initialized = False
                        self.last_bbox = None
            
        return self.last_bbox

    def calculate_crop_box(self, frame, mode="pan_only"):
        img_h, img_w, _ = frame.shape
        bbox = self.detect_and_track_person(frame)
        aspect_ratio = 9 / 16
        
        if bbox is None:
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
                crop_h = img_h
                crop_w = int(img_h * aspect_ratio)
                center_x = person_cx
                center_y = img_h / 2
            else: # pan_zoom
                target_person_ratio = 0.55
                crop_h = int(person_h / target_person_ratio)
                crop_h = min(img_h, max(int(img_h * 0.4), crop_h)) 
                crop_w = int(crop_h * aspect_ratio)
                center_x = person_cx
                center_y = person_cy

        if self.prev_center_x is None:
            self.prev_center_x = center_x
            self.prev_center_y = center_y
            self.prev_crop_w = crop_w
            self.prev_crop_h = crop_h
        
        easing_factor = 0.06
        scx = self.prev_center_x + easing_factor * (center_x - self.prev_center_x)
        scy = self.prev_center_y + easing_factor * (center_y - self.prev_center_y)
        scw = int(self.prev_crop_w + easing_factor * (crop_w - self.prev_crop_w))
        sch = int(self.prev_crop_h + easing_factor * (crop_h - self.prev_crop_h))

        self.prev_center_x = scx
        self.prev_center_y = scy
        self.prev_crop_w = scw
        self.prev_crop_h = sch

        cx1 = int(scx - scw / 2)
        cy1 = int(scy - sch / 2)
        
        if cx1 < 0: cx1 = 0
        if cy1 < 0: cy1 = 0
        if cx1 + scw > img_w: cx1 = img_w - scw
        if cy1 + sch > img_h: cy1 = img_h - sch

        scw = (scw // 2) * 2
        sch = (sch // 2) * 2
        return cx1, cy1, scw, sch
