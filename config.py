import os

# 🧭 强制开启 PyTorch 的 Mac MPS 显卡算力自动 fallback 机制
# 完美解决 torchvision::nms 算子在部分 Mac 显卡上未实现的 NotImplementedError
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Target Video Settings (9:16 standard Vertical Format)
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = 30

# YOLO Detection & Tracking Settings
YOLO_MODEL = "yolov8n.pt"  # Ultra-fast light model
PERSON_CLASS_ID = 0        # MS COCO class ID for 'person'
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence to detect person

# Tracking Smoothing Settings
SMOOTHING_WINDOW_SIZE = 15 # Moving average frame window
MOTION_THRESHOLD_RATIO = 0.02 # If crop center movement is smaller than 2% of width, don't move camera

# Face Mesh & Sticker Masking Settings
MASK_STICKER_PATH = os.path.join(os.path.dirname(__file__), "assets", "sticker.png")
# Scale multiplier for the sticker relative to face width
STICKER_SCALE_FACTOR = 1.6  
# Landmark indexes for eye corner measurement (used for face size calculation)
LEFT_EYE_CORNER_IDX = 263
RIGHT_EYE_CORNER_IDX = 33
NOSE_TIP_IDX = 4

# Hardware Acceleration Setup for macOS
# h264_videotoolbox uses Apple Silicon / Intel Mac hardware encoder
FFMPEG_VIDEO_ENCODER = "h264_videotoolbox"  
FFMPEG_DEFAULT_CRF = 23
FFMPEG_DEFAULT_PRESET = "fast"
