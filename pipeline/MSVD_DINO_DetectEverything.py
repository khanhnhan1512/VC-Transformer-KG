import os
import json
import time
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.ops as ops
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load Grounding DINO
MODEL_ID = "IDEA-Research/grounding-dino-base"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)

# --------------------------------------------
# Keyframe extraction
# --------------------------------------------
def extract_keyframes(video_path: str):
    """Return a list of PIL.Image corresponding to keyframes in the video."""
    keyframes = []
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        if frame.key_frame:
            img = frame.to_image()
            keyframes.append(img)
    container.close()
    return keyframes

# --------------------------------------------
# Detect Everything (generic prompt)
# --------------------------------------------
def detect_everything_in_frame(
    image_pil: Image.Image,
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    device: str,
    prompt: str = "all objects.",
    box_threshold: float = 0.2,
    text_threshold: float = 0.1,
):
    """Run Grounding DINO with a broad prompt to return all objects above thresholds."""
    inputs = processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],
    )[0]

    if len(results["scores"]) == 0:
        return []

    # Apply NMS to prune overlaps
    keep_indices = ops.nms(results["boxes"], results["scores"], iou_threshold=0.5)

    filtered = []
    for idx in keep_indices:
        idx = idx.item()
        filtered.append(
            {
                "label": results["labels"][idx],
                "score": results["scores"][idx].item(),
                "box": results["boxes"][idx].tolist(),
            }
        )
    return filtered

# --------------------------------------------
# Batch processing over dataset
# --------------------------------------------
OUTPUT_PATH = "../data/MSVD_raw/MSVD_DinoObjects_Everything.hdf5"
TIME_LIMIT_SECONDS = 20 * 3600 - 300  # 20 hours minus safety margin

video_paths = sorted(Path("../data/MSVD_raw/raw_video/").glob("*.avi"))
keyframe_df = pd.read_csv("../data/MSVD_raw/MSVD_keyframe_counts.csv")
vid_to_keyframe_count = dict(zip(keyframe_df.video_id, keyframe_df.keyframe_count))

# Resume support
processed_ids = set()
if os.path.exists(OUTPUT_PATH):
    try:
        with h5py.File(OUTPUT_PATH, "r") as f:
            processed_ids = set(f.keys())
        print(f"Found existing checkpoint with {len(processed_ids)} videos.")
    except OSError:
        print("Existing HDF5 is unreadable; a new file will be created.")

videos_to_process = [p for p in video_paths if p.stem not in processed_ids]
print(f"Remaining videos to process: {len(videos_to_process)}")

if len(videos_to_process) == 0:
    print("All videos are already processed.")
else:
    start_time = time.time()
    stop_flag = False
    pbar = tqdm(videos_to_process, desc="Processing")

    for vp in pbar:
        elapsed_time = time.time() - start_time
        if elapsed_time > TIME_LIMIT_SECONDS:
            print("\nTime limit reached; stopping to keep checkpoint safe.")
            stop_flag = True
            break

        video_id = vp.stem

        # Extract keyframes
        frames = extract_keyframes(vp)
        expected_count = vid_to_keyframe_count.get(video_id, 0)
        if expected_count and len(frames) != expected_count:
            print(f"Warning: keyframe count mismatch for {video_id} (expected {expected_count}, got {len(frames)}).")

        # Run detection per frame
        video_objects = []
        for frame_img in frames:
            detected_objs = detect_everything_in_frame(
                frame_img,
                PROCESSOR,
                MODEL,
                DEVICE,
                prompt="all objects.",
            )
            video_objects.append(detected_objs)

        # Serialize and save immediately
        dt = h5py.string_dtype(encoding="utf-8")
        frame_json_strings = [json.dumps(objs) for objs in video_objects]

        try:
            with h5py.File(OUTPUT_PATH, "a") as hf:
                if video_id in hf:
                    del hf[video_id]
                hf.create_dataset(video_id, data=np.array(frame_json_strings, dtype=object), dtype=dt)
        except Exception as e:  # noqa: BLE001
            print(f"\nError while saving {video_id}: {e}")
            continue

    if stop_flag:
        print("Paused; re-run to continue from the next video.")
    else:
        print("Finished current batch.")
