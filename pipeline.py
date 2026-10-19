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
    # 🚀 【神级性能优化：将本地 Homebrew 路径提为最高优先级！】
    # 本地编译包 100% 携带 VideoToolbox，可完美启用 AMD 独立显卡硬解，速度暴增！
    mac_fallbacks = [
        "/opt/homebrew/bin/ffmpeg",   # 1. Apple Silicon Mac (M1/M2/M3) 默认 Homebrew 路径
        "/usr/local/bin/ffmpeg",      # 2. Intel Mac 默认 Homebrew 路径
        project_bin,                  # 3. 项目内置的 static 编译包
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
    def __init__(self, input_path, output_path, tracking_mode="pan_only", select_index=None):
        self.input_path = input_path
        self.output_path = output_path
        self.tracking_mode = tracking_mode
        self.ffmpeg_path = get_ffmpeg_path()
        
        # 初始化追踪与人脸挡脸遮罩，并将外部 select_index 透传给 Tracker
        self.tracker = ReframeTracker(select_index=select_index)
        self.masker = FaceMasker()

    def _detect_scenes(self):
        """利用 PySceneDetect 进行镜头边界检测。"""
        print("🎬 正在分析视频场景切换中...")
        try:
            scene_list = detect(self.input_path, ContentDetector(threshold=27.0))
            scene_starts = [scene[0].get_frames() for scene in scene_list]
            print(f"🎬 检测完成！共发现 {len(scene_list)} 个镜头场景。")
            return scene_starts
        except Exception as e:
            print(f"⚠️ 场景检测失败 (使用常规首帧重置): {e}")
            return [0]

    def process_video(self):
        """执行完整的视频自动处理流水线"""
        if not os.path.exists(self.input_path):
            print(f"❌ 找不到输入视频: {self.input_path}")
            return False

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print("❌ 无法打开输入视频。")
            return False

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if orig_fps <= 0 or isnan(orig_fps):
            orig_fps = TARGET_FPS

        print(f"📹 视频加载成功: 尺寸={orig_width}x{orig_height}, 帧率={orig_fps:.2f}, 总帧数={total_frames}")

        scene_starts = self._detect_scenes()

        temp_silent_video = os.path.join(
            os.path.dirname(self.output_path), 
            f"temp_silent_{os.path.basename(self.output_path)}"
        )

        # 智能编码器检测与自适应切换
        encoder = FFMPEG_VIDEO_ENCODER
        if not check_encoder_support(self.ffmpeg_path, encoder):
            print(f"🧭 军师检测：当前 FFmpeg 静态包不支持 macOS 显卡硬解编码器 [{encoder}]。")
            print("   已为您自动、无缝切换为全兼容、超高清 CPU 编码器 [libx264]！")
            encoder = "libx264"
        else:
            print(f"🟢 [GPU ACCELERATED] 成功激活 macOS 原生显卡级 H.264 硬件硬解码器 [{encoder}]！")

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
            "-b:v", "4500k",
            "-pix_fmt", "yuv420p"
        ]

        if encoder == "libx264":
            ffmpeg_cmd.extend(["-preset", FFMPEG_DEFAULT_PRESET, "-crf", str(FFMPEG_DEFAULT_CRF)])
            
        ffmpeg_cmd.append(temp_silent_video)

        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        frame_idx = 0
        print("🚀 开始 AI 跟踪与贴纸渲染流水线...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx in scene_starts:
                    self.tracker.reset_tracking()
                
                # 1. 追踪我，计算 4K 原始帧中的 9:16 裁剪框坐标
                crop_x, crop_y, crop_w, crop_h = self.tracker.calculate_crop_box(
                    frame, mode=self.tracking_mode
                )
                
                # 2. 先执行 9:16 物理裁剪
                cropped_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                
                # 3. 将裁剪框等比缩放到标准的 1080x1920 视口分辨率
                resized_frame = cv2.resize(
                    cropped_frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC
                )
                
                all_candidates = self.tracker.all_candidates
                my_bbox = self.tracker.last_bbox
                
                # 4. 视口映射与局部 ROI 高斯羽化模糊
                final_frame = self.masker.apply_face_masks(
                    resized_frame, all_candidates, my_bbox, (crop_x, crop_y, crop_w, crop_h)
                )
                
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

        # 🚀 替换为更标准且全兼容的 1:a? 可选映射，彻底消灭 exit status 254 报错
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
            "-map", "1:a?",  # 🧭 标准可选匹配符，无音轨时自动不报错
            self.output_path
        ]
        
        try:
            # 捕获日志，方便调试
            res = subprocess.run(merge_audio_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"🎯 最终视频已大功告成！保存路径: {self.output_path}")
            if os.path.exists(temp_silent_video):
                os.remove(temp_silent_video)
            return True
        except Exception as e:
            # 🚀 【安全无缝退回机制】：若因 FFmpeg 静态包编译冲突或视频无音轨失败，直接把静音成品拷贝为最终输出，100% 确保导出大功告成！
            print(f"⚠️ 原始音频合并失败，正在执行安全降级，为您保存成品视频。错误: {e}")
            import shutil
            try:
                shutil.copy(temp_silent_video, self.output_path)
                if os.path.exists(temp_silent_video):
                    os.remove(temp_silent_video)
                print(f"🎯 降级成功！静音版成品视频已安全保存至: {self.output_path}")
                return True
            except Exception as se:
                print(f"❌ 降级处理失败: {se}")
                return False

def isnan(x):
    return x != x
