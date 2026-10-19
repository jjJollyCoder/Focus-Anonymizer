import os
import sys
import argparse
import cv2
import numpy as np
from pipeline import VideoPipeline
from config import MASK_STICKER_PATH

def generate_default_sticker():
    """极客暖心设计：缺省默认贴纸自动生成"""
    assets_dir = os.path.dirname(MASK_STICKER_PATH)
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    if os.path.exists(MASK_STICKER_PATH):
        return
    print("🧭 军师提醒：未检测到人脸贴纸，正在自动为您生成精美贴纸...")
    size = 400
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = 180
    cv2.circle(img, center, radius, (130, 220, 255, 230), -1) 
    cv2.circle(img, center, radius, (20, 160, 230, 255), 8)   
    cv2.circle(img, (140, 160), 20, (30, 40, 50, 255), -1)    
    cv2.circle(img, (260, 160), 20, (30, 40, 50, 255), -1)    
    axes = (60, 40)
    cv2.ellipse(img, (200, 240), axes, 0, 0, 180, (30, 40, 50, 255), 10)
    cv2.imwrite(MASK_STICKER_PATH, img)

def main():
    parser = argparse.ArgumentParser(
        description="🧭 Focus-Anonymizer - 极客短视频 100倍速自动裁剪与高斯模糊工作台"
    )
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="原视频文件路径 (.mp4 / .mov)"
    )
    parser.add_argument(
        "-o", "--output", 
        help="输出的 9:16 竖屏视频路径 (默认自动加 _vertical.mp4 后缀)"
    )
    parser.add_argument(
        "-m", "--mode", 
        choices=["pan_only", "pan_zoom"], 
        default="pan_only",
        help="追踪模式：pan_only (仅横向平移，最稳定，画质高); pan_zoom (居中锁定，远近稳定)"
    )
    parser.add_argument(
        "-s", "--select-index",
        type=int,
        default=None,
        help="在多人同框首帧中，强力指定锁定主体的编号 [0, 1, 2...] (无头 Headless/Agent 模式下跳过控制台输入)"
    )
    
    args = parser.parse_args()
    
    generate_default_sticker()
    
    input_path = os.path.abspath(args.input)
    if not args.output:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_vertical.mp4"
    else:
        output_path = os.path.abspath(args.output)
        
    print("=" * 60)
    print("🧭 智能短视频处理流水线启动中...")
    print(f"   输入视频: {input_path}")
    print(f"   输出视频: {output_path}")
    print(f"   AI 模式:  {args.mode} (支持 Apple Silicon Metal 硬件加速)")
    if args.select_index is not None:
        print(f"   无头指认: 强制指定追踪候选人编号为 [{args.select_index}]")
    print("=" * 60)
    
    # 将 select_index 顺畅透传给 VideoPipeline
    pipeline = VideoPipeline(input_path, output_path, tracking_mode=args.mode, select_index=args.select_index)
    success = pipeline.process_video()
    
    if success:
        print("\n🎉 处理成功！现在您可以 AirDrop 到 iPhone 手机上发布啦！")
        sys.exit(0)
    else:
        print("\n❌ 处理失败，请检查上面输出的错误排查信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()
