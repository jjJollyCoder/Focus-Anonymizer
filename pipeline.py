import os
import subprocess
import cv2
import numpy as np
from scenedetect import detect, ContentDetector

from config import *
from reframe import ReframeTracker
from mask import FaceMasker

def get_ffmpeg_path():
    """自动探测项目内置的 FFmpeg 静态二进制包、Mac 的 Homebrew 默认路径以及系统标准路径"""
    project_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "ffmpeg")
    mac_fallbacks = [
        project_bin,                  # 1. 项目内置的高性能 static 编译包（优先首选）
        "/opt/homebrew/bin/ffmpeg",   # 2. Apple Silicon Mac (M1/M2/M3) 默认 Homebrew 路径
        "/usr/local/bin/ffmpeg",      # 3. Intel Mac 默认 Homebrew 路径
        "ffmpeg"                      # 4. 兜底全局路径
    ]
    for path in mac_fallbacks:
        if path != "ffmpeg" and os.path.exists(path):
            return path
    return "ffmpeg"

def check_encoder_support(ffmpeg_path, encoder_name):
    """检查当前的 FFmpeg 二进制包是否支持特定的编码器"""
    try:
        res = subprocess.run([ffmpeg_path, "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return encoder_name in res.stdout
    except Exception:
        return False

class VideoPipeline:
    def __init__(self, input_path, output_path, tracking_mode="pan_only"):
        self.input_path = input_path
        self.output_path = output_path
        self.tracking_mode = tracking_mode
        self.ffmpeg_path = get_ffmpeg_path()
        
        # 初始化追踪与人脸挡脸遮罩
        self.tracker = ReframeTracker()
        self.masker = FaceMasker()

    def _detect_scenes(self):
        """
        利用 PySceneDetect 进行镜头边界检测。
        返回所有场景的起始帧列表。
        """
        print("🎬 正在分析视频场景切换中...")
        try:
            # 使用内容阈值探测器检测镜头切换
            scene_list = detect(self.input_path, ContentDetector(threshold=27.0))
            scene_starts = [scene[0].get_frames() for scene in scene_list]
            print(f"🎬 检测完成！共发现 {len(scene_list)} 个镜头场景。")
            return scene_starts
        except Exception as e:
            print(f"⚠️ 场景检测失败 (使用常规首帧重置): {e}")
            return [0]

    def process_video(self):
        """
        执行完整的视频自动处理流水线：
        1. 场景检测
        2. 帧读取 -> 追踪检测 -> 人脸贴纸 -> 9:16裁剪 -> 尺寸拉伸 -> FFmpeg管道写入
        3. 音频合并与导出
        """
        if not os.path.exists(self.input_path):
            print(f"❌ 找不到输入视频: {self.input_path}")
            return False

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print("❌ 无法打开输入视频。")
            return False

        # 获取视频原始属性
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if orig_fps <= 0 or isnan(orig_fps):
            orig_fps = TARGET_FPS

        print(f"📹 视频加载成功: 尺寸={orig_width}x{orig_height}, 帧率={orig_fps:.2f}, 总帧数={total_frames}")

        # 1. 场景切割探测
        scene_starts = self._detect_scenes()

        # 定义临时静音视频路径
        temp_silent_video = os.path.join(
            os.path.dirname(self.output_path), 
            f"temp_silent_{os.path.basename(self.output_path)}"
        )

        # 2. 智能编码器预检与自适应切换
        # 检测当前的 FFmpeg 静态二进制包是否支持 Mac 硬件加速硬解编码器
        encoder = FFMPEG_VIDEO_ENCODER
        if not check_encoder_support(self.ffmpeg_path, encoder):
            print(f"🧭 军师检测：当前 FFmpeg 静态包不支持 macOS 显卡硬解编码器 [{encoder}]。")
            print("   已为您自动、无缝切换为全兼容、超高清 CPU 编码器 [libx264]！")
            encoder = "libx264"

        # 通过 rawvideo 像素管道接收 Python 推送的高能处理帧
        ffmpeg_cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
            "-r", str(orig_fps),
            "-i", "pipe:0",
            "-c:v", encoder,
            "-b:v", "4500k",               # 4.5M 码率，确保高质高清，3分钟约 100MB
            "-pix_fmt", "yuv420p"          # 抖音支持的最稳色域格式
        ]

        # 如果是 libx264，注入标准的高画质、高效率优化参数
        if encoder == "libx264":
            ffmpeg_cmd.extend(["-preset", FFMPEG_DEFAULT_PRESET, "-crf", str(FFMPEG_DEFAULT_CRF)])
            
        ffmpeg_cmd.append(temp_silent_video)

        # 启动 FFmpeg 子进程
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        frame_idx = 0
        print("🚀 开始 AI 跟踪与贴纸渲染流水线...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检查是否为新场景的开始，重置追踪平滑器防止穿镜头跳跃
                if frame_idx in scene_starts:
                    self.tracker.reset_tracking()
                
                # 1. 追踪我，计算 4K 原始帧中的 9:16 裁剪框坐标 (cx, cy, cw, ch)
                crop_x, crop_y, crop_w, crop_h = self.tracker.calculate_crop_box(
                    frame, mode=self.tracking_mode
                )
                
                # 2. 先执行 9:16 物理裁剪（削减 75% 的多余像素计算量）
                cropped_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                
                # 3. 将裁剪框等比缩放到标准的 1080x1920 视口分辨率（Viewport）
                resized_frame = cv2.resize(
                    cropped_frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC
                )
                
                # 4. 提取当前帧 4K 下所有人的检测框，以及我本人的 tracker 框
                all_candidates = self.tracker.all_candidates
                my_bbox = self.tracker.last_bbox
                
                # 5. 【降维打击】视口映射与局部 ROI 模糊（自动裁掉视口外的人，并对在内的人进行毫秒级局部模糊）
                final_frame = self.masker.apply_face_masks(
                    resized_frame, all_candidates, my_bbox, (crop_x, crop_y, crop_w, crop_h)
                )
                
                # 6. 推送 raw 像素数据到 FFmpeg 子进程管道
                try:
                    ffmpeg_process.stdin.write(final_frame.tobytes())
                except IOError:
                    print("❌ FFmpeg 管道意外断开，停止渲染。")
                    break
                
                frame_idx += 1
                if frame_idx % 50 == 0:
                    percent = (frame_idx / total_frames) * 100
                    print(f"📊 已处理帧数: {frame_idx}/{total_frames} ({percent:.1f}%)")

        finally:
            cap.release()
            if ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()
            ffmpeg_process.wait()

        print("🎉 视频处理完成。正在合并原始音频...")

        # 3. 将原始音轨高保真合并到裁剪后的视频中 (采用 map 可选映射防出错)
        merge_audio_cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", temp_silent_video,
            "-i", self.input_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-ar", "44100",
            "-ab", "128k",
            "-map", "0:v:0",
            "-map", "1:a:0?",  # ? 代表即使没有音频也成功，安全第一
            self.output_path
        ]
        
        try:
            subprocess.run(merge_audio_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"🎯 最终视频已大功告成！保存路径: {self.output_path}")
            # 清理临时无声视频
            if os.path.exists(temp_silent_video):
                os.remove(temp_silent_video)
            return True
        except Exception as e:
            print(f"⚠️ 音频合并失败，静音版本视频保存在: {temp_silent_video}。错误: {e}")
            return False

def isnan(x):
    return x != x
