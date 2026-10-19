---
name: focus-anonymizer
display_name: Focus-Anonymizer 智能人像运镜与隐私模糊
description: 智能将横屏视频切为 9:16 竖屏，对旁人头部进行高斯模糊羽化，黄金居中锁定主体人物。
description_zh: 智能将横屏视频切为 9:16 竖屏，对旁人头部进行高斯模糊羽化，黄金居中锁定主体人物。
description_en: Focus-Anonymizer 100x speed smooth Lerp cinematic auto-reframe with local ROI gaussian feathering face-blur.
category: video
version: 1.0.0
author: jjJollyCoder
---

# 技能定义

当用户需要对一段亲子、跳舞、群聚等多人同框视频进行 9:16 竖屏裁剪、主角居中追踪、并对旁人进行挡脸隐私遮罩时，自动唤起此技能。

## 执行步骤

1. **原视频确认**：
   - 识别用户输入的视频文件。
   - 如果用户没有指定路径，提示用户将视频放入当前技能目录下的 `input_videos/` 文件夹中。

2. **调用本地脚本运行**：
   - 唤起本地命令行，进入当前技能所在目录：`cd /Users/jojo/Downloads/short_video_automation`。
   - 使用系统自带的 `bash` 执行 `./douyin` 快捷脚本开始运行：
     ```bash
     ./douyin
     ```

3. **辅助用户选人指认**：
   - 脚本运行中，当控制台由于 Scheme 2 机制弹出候选人预览图 `first_frame_select.png` 并询问指认编号时，将编号选择和提示完整转述并呈现给用户。
   - 接收用户输入的数字，写回控制台输入管道中。

4. **无缝结果交付**：
   - 当终端日志输出 `🎯 最终视频已大功告成！` 时，检测 `output_videos/` 目录下新生成的视频文件。
   - 将该竖屏视频直接作为任务交付产物展示在右侧结果区，并温馨提示用户可以通过右键隔空投送 (AirDrop) 传回 iPhone 发布。
