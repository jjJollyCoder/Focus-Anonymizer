import os
import sys
import subprocess
from typing import Optional
from mcp.server.fastmcp import FastMCP

# 初始化全球统一的 Model Context Protocol 服务器
mcp = FastMCP("Focus-Anonymizer")

@mcp.tool()
def reframe_and_blur_video(
    video_path: str, 
    mode: str = "pan_only", 
    select_index: Optional[int] = None
) -> str:
    """
    Run the Focus-Anonymizer pipeline to convert a horizontal video to a 9:16 vertical 
    video, smoothly tracking the main subject (using Lerp easing) and applying 
    locally-optimized, edge-feathered Gaussian blur to all other bystanders' faces.
    
    Arguments:
        video_path: The absolute local file path to the input horizontal video (.mp4 or .mov).
        mode: The tracking mode. Choose 'pan_only' (stable horizontal tracking, default) or 'pan_zoom' (auto-zooming and sizing target).
        select_index: Optional. The index of the person to lock (0, 1, 2...). If running inside an AI Agent, this bypasses the interactive CLI pop-up.
        
    Returns:
        A status message with the processed video output path or error logs.
    """
    if not os.path.isabs(video_path):
        return f"Error: Please provide an absolute file path for 'video_path'. Received: {video_path}"
        
    if not os.path.exists(video_path):
        return f"Error: Input video path does not exist: {video_path}"
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_bin = "/Users/jojo/.workbuddy/binaries/python/envs/venv_3.9/bin/python" # Mac 默认虚拟环境路径
    if not os.path.exists(python_bin):
        python_bin = "python3" # 跨平台 fallback
        
    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_vertical.mp4"
    
    # 组装命令行
    cmd = [python_bin, os.path.join(script_dir, "main.py"), "-i", video_path, "-o", output_path, "-m", mode]
    if select_index is not None:
        cmd.extend(["--select-index", str(select_index)])
        
    try:
        result = subprocess.run(
            cmd, 
            cwd=script_dir, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return f"🎉 Success! The processed video has been saved to: {output_path}\nLogs:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"❌ Execution failed.\nReturn code: {e.returncode}\nError:\n{e.stderr}\nLogs:\n{e.stdout}"

if __name__ == "__main__":
    mcp.run()
