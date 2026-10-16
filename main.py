import os
import sys
import argparse
import cv2
import numpy as np
from pipeline import VideoPipeline
from config import MASK_STICKER_PATH

def generate_default_sticker():
    """
    极客暖心设计：
    如果用户没有提供 sticker.png，自动在 assets 目录下生成一个精致、半透明的隐私保护“小黄脸遮脸贴纸”。
    避免程序报错，实现开箱即用。
    """
    assets_dir = os.path.dirname(MASK_STICKER_PATH)
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        
    if os.path.exists(MASK_STICKER_PATH):
        return

    print("🧭 军师提醒：未检测到人脸贴纸，正在自动为您生成精美的‘极客隐私保护小黄脸贴纸’...")
    
    # 创建一个 400x400 的透明底 RGBA 图像
    size = 400
    img = np.zeros((size, size, 4), dtype=np.uint8)
    
    # 画大圆脸（黄色，半透明 Alpha = 220）
    center = (size // 2, size // 2)
    radius = 180
    cv2.circle(img, center, radius, (130, 220, 255, 230), -1) # 温暖小黄圆脸
    cv2.circle(img, center, radius, (20, 160, 230, 255), 8)   # 深色边框
    
    # 画可爱的卡哇伊眯眯眼
    cv2.circle(img, (140, 160), 20, (30, 40, 50, 255), -1)    # 左眼
    cv2.circle(img, (260, 160), 20, (30, 40, 50, 255), -1)    # 右眼
    
    # 画快乐的微笑嘴巴
    axes = (60, 40)
    angle = 0
    startAngle = 0
    endAngle = 180
    color = (30, 40, 50, 255)
    thickness = 10
    cv2.ellipse(img, (200, 240), axes, angle, startAngle, endAngle, color, thickness)
    
    # 保存贴纸
    cv2.imwrite(MASK_STICKER_PATH, img)
    print(f"✨ 默认遮挡贴纸已成功生成至: {MASK_STICKER_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="🧭 极客带风控视频自动化工具栈 - 钉钉视频横转竖 & 人像追踪 & 动态挡脸"
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
    
    args = parser.parse_args()
    
    # 1. 自动生成默认贴纸，保障零配置运行
    generate_default_sticker()
    
    # 2. 处理输出路径
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
    print("=" * 60)
    
    pipeline = VideoPipeline(input_path, output_path, tracking_mode=args.mode)
    success = pipeline.process_video()
    
    if success:
        print("\n🎉 处理成功！现在您可以 AirDrop 到 iPhone 手机上，用抖音客户端进行最终的美化与发布啦！")
        sys.exit(0)
    else:
        print("\n❌ 处理失败，请检查上面输出的错误排查信息。")
        sys.exit(1)


if __name__ == "__main__":
    main()
