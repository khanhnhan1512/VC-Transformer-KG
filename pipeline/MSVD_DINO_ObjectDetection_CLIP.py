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

# Import CLIP
import clip

# ------------------ CONFIGURATION ------------------
# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths (Bạn hãy kiểm tra lại đường dẫn chính xác trên server)
MODEL_ID = "IDEA-Research/grounding-dino-base"
ENTITIES_PATH = "./OPENKE_file/entities.txt"
OUTPUT_PATH = "../data/MSVD_raw/MSVD_DinoObjects_CLIP_filtered.hdf5"
VIDEO_DIR = "../data/MSVD_raw/raw_video/"
KEYFRAME_META = "../data/MSVD_raw/MSVD_keyframe_counts.csv"

# Hyperparameters
CLIP_MODEL_NAME = "ViT-B/32"  # Nhẹ và nhanh, đủ dùng để lọc nhãn
CLIP_TOP_K = 35               # Số lượng nhãn tối đa đưa vào DINO cho mỗi ảnh (giảm xuống 20-30 nếu OOM)
BOX_THRESHOLD = 0.1 # 0.35
TEXT_THRESHOLD = 0.1 # 0.25

# Time limit for cluster jobs (e.g., 20 hours)
TIME_LIMIT_SECONDS = 20 * 3600 - 300 

# ------------------ MODEL LOADING ------------------

print("Loading Grounding DINO...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

print(f"Loading CLIP ({CLIP_MODEL_NAME})...")
clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=device)

# ------------------ HELPER FUNCTIONS ------------------

def load_labels_from_file(file_path: str):
    """Load labels và chuẩn hóa text."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Entities file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        # Loại bỏ dòng trống và chuẩn hóa lowercase
        labels = [line.strip().lower() for line in f if line.strip()]
    return sorted(list(set(labels))) # Deduplicate & sort

def precompute_text_features(labels, batch_size=256):
    """
    Mã hóa toàn bộ nhãn thành vector CLIP Text Features.
    Chạy 1 lần duy nhất khi bắt đầu.
    """
    print(f"Pre-computing CLIP features for {len(labels)} labels...")
    all_features = []
    
    # Batching để tránh OOM nếu danh sách nhãn quá lớn
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i : i + batch_size]
        # Tạo prompt giả lập ngữ cảnh: "a photo of a {label}"
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in batch_labels]).to(device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True) # Normalize
            all_features.append(text_features)
            
    return torch.cat(all_features, dim=0)

def extract_keyframes(video_path: str):
    """Trích xuất I-frames từ video."""
    keyframes = []
    try:
        container = av.open(str(video_path))
        for frame in container.decode(video=0):
            if frame.key_frame:
                keyframes.append(frame.to_image()) # Return PIL Image
        container.close()
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
    return keyframes

def get_relevant_labels(image_pil, all_labels, text_features_bank, top_k):
    """
    Dùng CLIP để lọc ra top_k nhãn phù hợp nhất với ảnh hiện tại.
    """
    # Preprocess ảnh cho CLIP
    image_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_feature = clip_model.encode_image(image_input)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        
        # Tính Cosine Similarity: (1, dim) @ (N_labels, dim).T -> (1, N_labels)
        similarity = (100.0 * image_feature @ text_features_bank.T).softmax(dim=-1)
        
        # Lấy top k
        values, indices = similarity[0].topk(k=min(top_k, len(all_labels)))
        
    # Map index về text label
    relevant_labels = [all_labels[idx] for idx in indices.cpu().numpy()]
    return relevant_labels

def detect_objects_optimized(image_pil, all_labels, text_features_bank, 
                             processor, model, device):
    """
    Pipeline chính: CLIP Filter -> DINO Detect -> NMS
    """
    # 1. Lọc nhãn bằng CLIP
    candidate_labels = get_relevant_labels(image_pil, all_labels, text_features_bank, CLIP_TOP_K)
    
    if not candidate_labels:
        return []

    # 2. Tạo prompt cho DINO
    # Lưu ý: Grounding DINO cần dấu chấm ngăn cách các label
    text_prompt = ". ".join(candidate_labels) + "."
    
    # 3. Chạy Grounding DINO
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Post-processing
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image_pil.size[::-1]],
    )[0]

    # 5. Global NMS (Non-Maximum Suppression)
    # Dù chỉ chạy 1 lần prompt, DINO vẫn có thể trả về các box chồng lấn
    if len(results["scores"]) == 0:
        return []

    keep = ops.nms(results["boxes"], results["scores"], iou_threshold=0.5)
    
    filtered_results = []
    for idx in keep:
        i = idx.item()
        filtered_results.append({
            "label": results["labels"][i],
            "score": float(results["scores"][i].item()),
            "box": results["boxes"][i].tolist() # [x1, y1, x2, y2]
        })
        
    return filtered_results

# ------------------ MAIN PIPELINE ------------------

def main():
    # 1. Load Data
    entities_list = load_labels_from_file(ENTITIES_PATH)
    print(f"Loaded {len(entities_list)} unique labels.")
    
    # 2. Precompute CLIP Embeddings (Chỉ làm 1 lần)
    label_features_bank = precompute_text_features(entities_list)
    print(f"Text features bank shape: {label_features_bank.shape}")

    # 3. Setup Video Processing
    if os.path.exists(KEYFRAME_META):
        keyframe_df = pd.read_csv(KEYFRAME_META)
        vid_to_keyframe_count = dict(zip(keyframe_df.video_id, keyframe_df.keyframe_count))
    else:
        print("Warning: Keyframe metadata file not found. Skipping validation.")
        vid_to_keyframe_count = {}

    video_paths = sorted(Path(VIDEO_DIR).glob("*.avi"))
    
    # 4. Checkpoint Logic
    processed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        try:
            with h5py.File(OUTPUT_PATH, "r") as f:
                processed_ids = set(f.keys())
            print(f"Found checkpoint. Already processed: {len(processed_ids)} videos.")
        except OSError:
            print("⚠️ Checkpoint file is corrupted or unreadable.")

    videos_to_process = [p for p in video_paths if p.stem not in processed_ids]
    print(f"Pending videos: {len(videos_to_process)}")

    if not videos_to_process:
        print("All videos processed. Exiting.")
        return

    # 5. Processing Loop
    start_time = time.time()
    stop_flag = False
    pbar = tqdm(videos_to_process, desc="Processing Videos")

    # Mở HDF5 ở chế độ append bên ngoài vòng lặp (hoặc mở/đóng từng file để an toàn hơn)
    # Để an toàn tối đa cho data khi crash, ta sẽ mở/đóng file trong vòng lặp (chậm hơn chút nhưng an toàn)
    
    for vp in pbar:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"\nTime limit reached ({elapsed/3600:.2f} h). Stopping safely.")
            stop_flag = True
            break

        vid_id = vp.stem
        frames = extract_keyframes(vp)
        
        # Validate frame count (optional)
        expected = vid_to_keyframe_count.get(vid_id, 0)
        if expected > 0 and len(frames) != expected:
            # Chỉ warning log, không crash
            pass 

        video_objects = []
        
        try:
            for frame_img in frames:
                detected = detect_objects_optimized(
                    frame_img, 
                    entities_list, 
                    label_features_bank,
                    processor, 
                    dino_model, 
                    device
                )
                video_objects.append(detected)
                
            # Save to HDF5
            dt = h5py.string_dtype(encoding="utf-8")
            frame_json_strings = [json.dumps(objs) for objs in video_objects]
            
            with h5py.File(OUTPUT_PATH, "a") as hf:
                if vid_id in hf:
                    del hf[vid_id] # Xóa nếu đã tồn tại (trường hợp chạy lại đè)
                
                hf.create_dataset(
                    vid_id,
                    data=np.array(frame_json_strings, dtype=object),
                    dtype=dt
                )
                
        except Exception as e:
            print(f"\nFailed to process/save {vid_id}: {e}")
            continue
            
        # Clean up memory occasionally if needed
        # torch.cuda.empty_cache() 

    if stop_flag:
        print("Paused. Re-run script to continue.")
    else:
        print("Batch processing completed successfully.")

if __name__ == "__main__":
    main()