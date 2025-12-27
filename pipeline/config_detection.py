"""
Configuration for Grounded SAM 2 Detection Pipeline
===================================================
Sử dụng Grounding DINO (HuggingFace) + SAM 2.1 Hiera Large cho MSVD dataset.
"""

import os
from pathlib import Path

# ==============================================================================
# DEVICE & ENVIRONMENT
# ==============================================================================
DEVICE = "cuda"  # "cuda" hoặc "cpu"
DTYPE = "bfloat16"  # "float32", "float16", "bfloat16" - bfloat16 tiết kiệm VRAM nhất

# ==============================================================================
# MODEL PATHS & CONFIGURATIONS
# ==============================================================================

# Grounding DINO (HuggingFace)
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# SAM 2.1 Hiera Large (Local)
# Download: wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ==============================================================================
# DETECTION THRESHOLDS
# ==============================================================================
BOX_THRESHOLD = 0.35      # Ngưỡng confidence cho bounding box (0.3-0.4 tối ưu)
TEXT_THRESHOLD = 0.25     # Ngưỡng similarity cho text matching
NMS_IOU_THRESHOLD = 0.5   # IoU threshold cho Non-Maximum Suppression

# ==============================================================================
# CLIP FILTERING (Optional - Lọc nhãn trước khi đưa vào DINO)
# ==============================================================================
USE_CLIP_FILTERING = True
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_TOP_K = 35           # Số nhãn tối đa sau khi lọc bằng CLIP

# ==============================================================================
# TEMPORAL TRACKING (SAM 2 Video Mode)
# ==============================================================================
RE_DETECTION_INTERVAL = 30  # Số frame giữa các lần re-detect bằng Grounding DINO
MAX_OBJECTS_PER_FRAME = 50  # Giới hạn số object tối đa mỗi frame
TRACKING_MEMORY_BANK_SIZE = 7  # Số frame lưu trong memory bank của SAM 2

# ==============================================================================
# BATCH & MEMORY SETTINGS (Tối ưu cho ổn định)
# ==============================================================================
BATCH_SIZE = 1            # Xử lý từng frame một để tối đa ổn định
FRAMES_PER_VIDEO_BATCH = 10  # Số frame xử lý trước khi clear cache
CLEAR_CACHE_INTERVAL = 5  # Clear CUDA cache sau mỗi N videos

# ==============================================================================
# DATA PATHS
# ==============================================================================
# Đường dẫn tương đối từ thư mục pipeline/
_BASE_DIR = Path(__file__).parent.parent

# Input
ENTITIES_PATH = _BASE_DIR / "OPENKE_file" / "entities.txt"
VIDEO_DIR = _BASE_DIR / "data" / "MSVD_raw-20251110T043013Z-1-001" / "MSVD_raw" / "raw_video"
KEYFRAME_META = _BASE_DIR / "data" / "MSVD_raw-20251110T043013Z-1-001" / "MSVD_raw" / "MSVD_keyframe_counts.csv"

# Output
OUTPUT_DIR = _BASE_DIR / "data" / "result"
OUTPUT_HDF5 = OUTPUT_DIR / "MSVD_GroundedSAM2_Objects.hdf5"
OUTPUT_TRACKING_HDF5 = OUTPUT_DIR / "MSVD_GroundedSAM2_Tracking.hdf5"

# ==============================================================================
# TIME LIMIT (Cho các job cluster như Kaggle/Colab)
# ==============================================================================
TIME_LIMIT_HOURS = 20
TIME_LIMIT_SECONDS = TIME_LIMIT_HOURS * 3600 - 300  # Trừ 5 phút buffer

# ==============================================================================
# LOGGING
# ==============================================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
SAVE_VISUALIZATION = False  # Lưu ảnh với bounding box để debug
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def ensure_directories():
    """Tạo các thư mục output nếu chưa tồn tại."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_VISUALIZATION:
        VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

def get_dtype_torch():
    """Chuyển đổi string dtype sang torch dtype."""
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(DTYPE, torch.bfloat16)

def print_config():
    """In ra cấu hình hiện tại."""
    print("=" * 60)
    print("GROUNDED SAM 2 DETECTION CONFIGURATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Grounding DINO: {GROUNDING_DINO_MODEL_ID}")
    print(f"SAM 2 Checkpoint: {SAM2_CHECKPOINT}")
    print(f"Box Threshold: {BOX_THRESHOLD}")
    print(f"Text Threshold: {TEXT_THRESHOLD}")
    print(f"Use CLIP Filtering: {USE_CLIP_FILTERING}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Re-detection Interval: {RE_DETECTION_INTERVAL} frames")
    print(f"Output: {OUTPUT_HDF5}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
    ensure_directories()
    print("\nPaths validation:")
    print(f"  Entities file exists: {ENTITIES_PATH.exists()}")
    print(f"  Video dir exists: {VIDEO_DIR.exists()}")
    print(f"  Keyframe meta exists: {KEYFRAME_META.exists()}")
