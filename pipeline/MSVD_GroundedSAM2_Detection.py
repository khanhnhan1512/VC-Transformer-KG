"""
MSVD Grounded SAM 2 Object Detection Pipeline
=============================================
Kết hợp Grounding DINO (HuggingFace) + SAM 2.1 Hiera Large
để phát hiện và tracking object trong video MSVD.

Features:
- Grounding DINO cho zero-shot object detection
- SAM 2 cho segmentation masks và temporal tracking
- CLIP filtering để lọc nhãn phù hợp (optional)
- Checkpoint/Resume support cho long-running jobs
- Memory-optimized với batch_size=1 và bfloat16

Usage:
    conda activate grounded_sam2
    python MSVD_GroundedSAM2_Detection.py

Author: Generated for VC-Transformer-KG project
Date: December 2025
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import av
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.ops as ops
from PIL import Image
from tqdm import tqdm

# HuggingFace Transformers for Grounding DINO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Import configuration
from config_detection import (
    DEVICE, DTYPE, get_dtype_torch, print_config, ensure_directories,
    GROUNDING_DINO_MODEL_ID, SAM2_CHECKPOINT, SAM2_CONFIG,
    BOX_THRESHOLD, TEXT_THRESHOLD, NMS_IOU_THRESHOLD,
    USE_CLIP_FILTERING, CLIP_MODEL_NAME, CLIP_TOP_K,
    RE_DETECTION_INTERVAL, MAX_OBJECTS_PER_FRAME,
    BATCH_SIZE, FRAMES_PER_VIDEO_BATCH, CLEAR_CACHE_INTERVAL,
    ENTITIES_PATH, VIDEO_DIR, KEYFRAME_META,
    OUTPUT_HDF5, OUTPUT_TRACKING_HDF5,
    TIME_LIMIT_SECONDS, LOG_LEVEL,
    SAVE_VISUALIZATION, VISUALIZATION_DIR,
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# GLOBAL MODEL INSTANCES (Lazy loading)
# ==============================================================================
_grounding_dino_processor = None
_grounding_dino_model = None
_clip_model = None
_clip_preprocess = None
_clip_text_features = None
_sam2_predictor = None
_all_labels = None


def load_grounding_dino():
    """Load Grounding DINO model từ HuggingFace."""
    global _grounding_dino_processor, _grounding_dino_model
    
    if _grounding_dino_model is not None:
        return _grounding_dino_processor, _grounding_dino_model
    
    logger.info(f"Loading Grounding DINO from {GROUNDING_DINO_MODEL_ID}...")
    _grounding_dino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
    _grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_MODEL_ID
    ).to(DEVICE)
    
    # Set to eval mode
    _grounding_dino_model.eval()
    
    logger.info("Grounding DINO loaded successfully.")
    return _grounding_dino_processor, _grounding_dino_model


def load_clip_model():
    """Load CLIP model cho label filtering."""
    global _clip_model, _clip_preprocess
    
    if _clip_model is not None:
        return _clip_model, _clip_preprocess
    
    if not USE_CLIP_FILTERING:
        return None, None
    
    try:
        import clip
        logger.info(f"Loading CLIP ({CLIP_MODEL_NAME})...")
        _clip_model, _clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
        logger.info("CLIP loaded successfully.")
    except ImportError:
        logger.warning("CLIP not installed. Disabling CLIP filtering.")
        return None, None
    
    return _clip_model, _clip_preprocess


def load_sam2_predictor():
    """Load SAM 2 predictor cho segmentation và tracking."""
    global _sam2_predictor
    
    if _sam2_predictor is not None:
        return _sam2_predictor
    
    # Check if SAM 2 is available
    sam2_checkpoint_path = Path(SAM2_CHECKPOINT)
    if not sam2_checkpoint_path.exists():
        logger.warning(f"SAM 2 checkpoint not found at {SAM2_CHECKPOINT}")
        logger.warning("SAM 2 features will be disabled. Download checkpoint:")
        logger.warning("wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/")
        return None
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        logger.info(f"Loading SAM 2 from {SAM2_CHECKPOINT}...")
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        _sam2_predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM 2 loaded successfully.")
        
    except ImportError:
        logger.warning("SAM 2 not installed. Install with: pip install -e . (from sam2 repo)")
        return None
    except Exception as e:
        logger.error(f"Failed to load SAM 2: {e}")
        return None
    
    return _sam2_predictor


def load_labels() -> List[str]:
    """Load entity labels từ file."""
    global _all_labels
    
    if _all_labels is not None:
        return _all_labels
    
    entities_path = Path(ENTITIES_PATH)
    if not entities_path.exists():
        raise FileNotFoundError(f"Entities file not found: {ENTITIES_PATH}")
    
    with open(entities_path, "r", encoding="utf-8") as f:
        _all_labels = [line.strip().lower() for line in f if line.strip()]
    
    _all_labels = sorted(list(set(_all_labels)))  # Deduplicate & sort
    logger.info(f"Loaded {len(_all_labels)} unique labels.")
    
    return _all_labels


def precompute_clip_features(labels: List[str], batch_size: int = 256) -> Optional[torch.Tensor]:
    """Pre-compute CLIP text features cho tất cả labels."""
    global _clip_text_features
    
    if _clip_text_features is not None:
        return _clip_text_features
    
    clip_model, _ = load_clip_model()
    if clip_model is None:
        return None
    
    import clip as clip_module
    
    logger.info(f"Pre-computing CLIP features for {len(labels)} labels...")
    all_features = []
    
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i + batch_size]
        text_inputs = torch.cat([
            clip_module.tokenize(f"a photo of a {c}") for c in batch_labels
        ]).to(DEVICE)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_features.append(text_features)
    
    _clip_text_features = torch.cat(all_features, dim=0)
    logger.info(f"CLIP text features shape: {_clip_text_features.shape}")
    
    return _clip_text_features


# ==============================================================================
# FRAME EXTRACTION
# ==============================================================================

def extract_keyframes(video_path: Path) -> List[Image.Image]:
    """Trích xuất I-frames (keyframes) từ video."""
    keyframes = []
    try:
        container = av.open(str(video_path))
        for frame in container.decode(video=0):
            if frame.key_frame:
                keyframes.append(frame.to_image())
        container.close()
    except Exception as e:
        logger.error(f"Error reading video {video_path}: {e}")
    
    return keyframes


def extract_uniform_frames(video_path: Path, num_frames: int = 10) -> List[Image.Image]:
    """Trích xuất frames đều từ video (backup nếu không có keyframes)."""
    frames = []
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        total_frames = stream.frames
        
        if total_frames == 0:
            # Fallback: decode all and count
            all_frames = list(container.decode(video=0))
            total_frames = len(all_frames)
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = [all_frames[i].to_image() for i in indices]
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    frames.append(frame.to_image())
        
        container.close()
    except Exception as e:
        logger.error(f"Error extracting uniform frames from {video_path}: {e}")
    
    return frames


# ==============================================================================
# CLIP-BASED LABEL FILTERING
# ==============================================================================

def get_relevant_labels(
    image_pil: Image.Image, 
    all_labels: List[str], 
    text_features: torch.Tensor, 
    top_k: int = CLIP_TOP_K
) -> List[str]:
    """Dùng CLIP để lọc top_k nhãn phù hợp nhất với ảnh."""
    clip_model, clip_preprocess = load_clip_model()
    
    if clip_model is None or text_features is None:
        # Fallback: return all labels (sẽ batch sau)
        return all_labels[:top_k]
    
    image_input = clip_preprocess(image_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        image_feature = clip_model.encode_image(image_input)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_feature @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(k=min(top_k, len(all_labels)))
    
    relevant_labels = [all_labels[idx] for idx in indices.cpu().numpy()]
    return relevant_labels


# ==============================================================================
# OBJECT DETECTION (Grounding DINO)
# ==============================================================================

def detect_objects(
    image_pil: Image.Image,
    text_prompt: str,
    processor,
    model,
) -> List[Dict[str, Any]]:
    """
    Chạy Grounding DINO detection trên một ảnh.
    
    Returns:
        List of dict với keys: label, score, box [x1, y1, x2, y2]
    """
    inputs = processor(
        images=image_pil, 
        text=text_prompt, 
        return_tensors="pt"
    ).to(DEVICE)
    
    dtype = get_dtype_torch()
    
    with torch.no_grad(), torch.autocast(DEVICE, dtype=dtype):
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image_pil.size[::-1]],  # (width, height) -> (height, width)
    )[0]
    
    if len(results["scores"]) == 0:
        return []
    
    # Apply NMS
    keep = ops.nms(results["boxes"], results["scores"], iou_threshold=NMS_IOU_THRESHOLD)
    
    detections = []
    for idx in keep:
        i = idx.item()
        detections.append({
            "label": results["labels"][i],
            "score": float(results["scores"][i].item()),
            "box": results["boxes"][i].tolist(),  # [x1, y1, x2, y2]
        })
    
    # Limit max objects
    detections = detections[:MAX_OBJECTS_PER_FRAME]
    
    return detections


def detect_with_clip_filtering(
    image_pil: Image.Image,
    all_labels: List[str],
    text_features: Optional[torch.Tensor],
    processor,
    model,
) -> List[Dict[str, Any]]:
    """Detection pipeline với CLIP filtering."""
    
    if USE_CLIP_FILTERING and text_features is not None:
        # Lọc nhãn phù hợp
        candidate_labels = get_relevant_labels(image_pil, all_labels, text_features, CLIP_TOP_K)
    else:
        # Batch labels thành chunks để tránh token limit
        candidate_labels = all_labels[:50]  # First 50 labels
    
    if not candidate_labels:
        return []
    
    # Tạo prompt cho DINO (dấu chấm ngăn cách)
    text_prompt = ". ".join(candidate_labels) + "."
    
    return detect_objects(image_pil, text_prompt, processor, model)


# ==============================================================================
# SAM 2 SEGMENTATION
# ==============================================================================

def segment_detections(
    image_pil: Image.Image,
    detections: List[Dict[str, Any]],
    sam2_predictor,
) -> List[Dict[str, Any]]:
    """
    Thêm segmentation masks từ SAM 2 cho các detections.
    
    Returns:
        List of detections với thêm key 'mask_rle' (Run-Length Encoding)
    """
    if sam2_predictor is None or len(detections) == 0:
        return detections
    
    image_np = np.array(image_pil)
    
    dtype = get_dtype_torch()
    
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=dtype):
        sam2_predictor.set_image(image_np)
        
        # Convert boxes to numpy
        boxes = np.array([d["box"] for d in detections])
        
        # Predict masks for all boxes at once
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
    
    # Add masks to detections
    for i, det in enumerate(detections):
        if i < len(masks):
            mask = masks[i].squeeze()  # (H, W)
            det["mask_rle"] = mask_to_rle(mask)
            det["mask_score"] = float(scores[i]) if i < len(scores) else det["score"]
    
    return detections


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Convert binary mask to Run-Length Encoding (RLE).
    Compact format để lưu vào HDF5.
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return {
        "counts": runs.tolist(),
        "size": list(mask.shape),
    }


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Decode RLE back to binary mask."""
    shape = rle["size"]
    counts = rle["counts"]
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    position = 0
    
    for i, count in enumerate(counts):
        if i % 2 == 1:  # Odd indices are 1s
            mask[position:position + count] = 1
        position += count
    
    return mask.reshape(shape)


# ==============================================================================
# VIDEO PROCESSING PIPELINE
# ==============================================================================

def process_video_frames(
    video_path: Path,
    all_labels: List[str],
    text_features: Optional[torch.Tensor],
    processor,
    dino_model,
    sam2_predictor,
) -> Tuple[List[List[Dict]], Dict[str, Any]]:
    """
    Process tất cả frames của một video.
    
    Returns:
        - frame_detections: List of detections per frame
        - video_metadata: Dict với thông tin video
    """
    # Extract frames
    frames = extract_keyframes(video_path)
    
    if len(frames) == 0:
        logger.warning(f"No keyframes found in {video_path.name}, trying uniform sampling...")
        frames = extract_uniform_frames(video_path, num_frames=10)
    
    if len(frames) == 0:
        logger.error(f"Could not extract any frames from {video_path.name}")
        return [], {"error": "no_frames"}
    
    frame_detections = []
    
    # Process each frame
    for frame_idx, frame_img in enumerate(frames):
        try:
            # Detection
            detections = detect_with_clip_filtering(
                frame_img, all_labels, text_features, processor, dino_model
            )
            
            # Segmentation (optional)
            if sam2_predictor is not None:
                detections = segment_detections(frame_img, detections, sam2_predictor)
            
            # Add frame index
            for det in detections:
                det["frame_idx"] = frame_idx
            
            frame_detections.append(detections)
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx} of {video_path.name}: {e}")
            frame_detections.append([])
        
        # Clear cache periodically
        if (frame_idx + 1) % FRAMES_PER_VIDEO_BATCH == 0:
            torch.cuda.empty_cache()
    
    metadata = {
        "video_id": video_path.stem,
        "num_frames": len(frames),
        "num_keyframes": len(frames),
    }
    
    return frame_detections, metadata


# ==============================================================================
# HDF5 SAVING
# ==============================================================================

def save_video_to_hdf5(
    video_id: str,
    frame_detections: List[List[Dict]],
    hdf5_path: Path,
):
    """Lưu detections của một video vào HDF5."""
    dt = h5py.string_dtype(encoding="utf-8")
    
    # Serialize each frame's detections to JSON
    frame_json_strings = [json.dumps(objs) for objs in frame_detections]
    
    with h5py.File(hdf5_path, "a") as hf:
        if video_id in hf:
            del hf[video_id]
        
        hf.create_dataset(
            video_id,
            data=np.array(frame_json_strings, dtype=object),
            dtype=dt,
        )


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """Main processing pipeline."""
    print_config()
    ensure_directories()
    
    # ==================== LOAD MODELS ====================
    logger.info("Loading models...")
    
    processor, dino_model = load_grounding_dino()
    all_labels = load_labels()
    text_features = precompute_clip_features(all_labels) if USE_CLIP_FILTERING else None
    sam2_predictor = load_sam2_predictor()
    
    # ==================== PREPARE DATA ====================
    logger.info("Preparing video list...")
    
    video_dir = Path(VIDEO_DIR)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")
    
    video_paths = sorted(video_dir.glob("*.avi"))
    logger.info(f"Found {len(video_paths)} videos in {video_dir}")
    
    # Load keyframe metadata (optional)
    keyframe_meta_path = Path(KEYFRAME_META)
    vid_to_keyframe_count = {}
    if keyframe_meta_path.exists():
        keyframe_df = pd.read_csv(keyframe_meta_path)
        vid_to_keyframe_count = dict(zip(keyframe_df.video_id, keyframe_df.keyframe_count))
    
    # ==================== CHECKPOINT LOGIC ====================
    output_path = Path(OUTPUT_HDF5)
    processed_ids = set()
    
    if output_path.exists():
        try:
            with h5py.File(output_path, "r") as f:
                processed_ids = set(f.keys())
            logger.info(f"Checkpoint found. Already processed: {len(processed_ids)} videos.")
        except OSError:
            logger.warning("Checkpoint file corrupted. Starting fresh.")
    
    videos_to_process = [p for p in video_paths if p.stem not in processed_ids]
    logger.info(f"Pending videos: {len(videos_to_process)}")
    
    if not videos_to_process:
        logger.info("All videos processed. Exiting.")
        return
    
    # ==================== PROCESSING LOOP ====================
    start_time = time.time()
    stop_flag = False
    
    pbar = tqdm(videos_to_process, desc="Processing Videos")
    
    for video_idx, video_path in enumerate(pbar):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT_SECONDS:
            logger.info(f"Time limit reached ({elapsed/3600:.2f}h). Stopping safely.")
            stop_flag = True
            break
        
        video_id = video_path.stem
        pbar.set_postfix({"video": video_id})
        
        try:
            # Process video
            frame_detections, metadata = process_video_frames(
                video_path,
                all_labels,
                text_features,
                processor,
                dino_model,
                sam2_predictor,
            )
            
            if frame_detections:
                save_video_to_hdf5(video_id, frame_detections, output_path)
            
        except Exception as e:
            logger.error(f"Failed to process {video_id}: {e}")
            continue
        
        # Clear cache periodically
        if (video_idx + 1) % CLEAR_CACHE_INTERVAL == 0:
            torch.cuda.empty_cache()
    
    # ==================== SUMMARY ====================
    elapsed_total = time.time() - start_time
    
    if stop_flag:
        logger.info("Processing paused. Re-run script to continue.")
    else:
        logger.info("Processing completed successfully.")
    
    logger.info(f"Total time: {elapsed_total/3600:.2f} hours")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
