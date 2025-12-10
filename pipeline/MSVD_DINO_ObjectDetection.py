import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import h5py
import av
import torch
import torchvision.ops as ops
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# --- Cấu hình thiết bị ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Model Grounding DINO ---
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def load_labels_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Đọc từng dòng, xóa khoảng trắng thừa và chuyển về chữ thường (quan trọng cho DINO)
        labels = [line.strip().lower() for line in f if line.strip()]
    return labels

# Load nhãn
entities_list = load_labels_from_file("./OPENKE_file/entities.txt") # Thay đường dẫn file của bạn vào đây
print(f"Loaded {len(entities_list)} labels: {entities_list[:5]}...")

##############################################
# Xác định số lượng khung hình chính (keyframes) trong video
###############################################

# Tạo một dictionary ánh xạ từ video_id đến số lượng khung hình chính (keyframes)
keyframe_df = pd.read_csv("../data/MSVD_raw/MSVD_keyframe_counts.csv")
vid_to_keyframe_count = {video_id: keyframe_count
                         for video_id, keyframe_count in zip(keyframe_df["video_id"], keyframe_df["keyframe_count"])}

##############################################
# Tách keyframes từ video dùng PyAV
##############################################

def extract_keyframes(video_path: str):
    """
    Return a list of PIL.Image corresponding to keyframes in the video.
    """
    keyframes = []
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        if frame.key_frame:
            img = frame.to_image()
            keyframes.append(img)
    container.close()
    return keyframes

##############################################
# Object Detection với Grounding DINO
##############################################
def detect_objects_in_frame(image_pil, labels_list, processor, model, device,
                            box_threshold=0.35, text_threshold=0.25, label_batch_size=50):
    """
    Chạy Grounding DINO trên 1 frame với danh sách nhãn rất dài (chia batch).
    Trả về: Danh sách các dict {'label': str, 'score': float, 'box': [x,y,w,h]}
    """

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = [] # Lưu text labels

    # Chia nhỏ danh sách nhãn để tránh lỗi token limit (256 tokens)
    # label_batch_size khoảng 50-80 là an toàn tùy độ dài từ
    num_batches = len(labels_list) // label_batch_size + (1 if len(labels_list) % label_batch_size != 0 else 0)

    # Helper chia batch
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    for batch_labels in chunks(labels_list, label_batch_size):
        # Tạo prompt: "cat. dog. car."
        text_prompt = ". ".join(batch_labels) + "."

        inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]]
        )[0]

        if len(results["scores"]) > 0:
            all_pred_boxes.append(results["boxes"])
            all_pred_scores.append(results["scores"])
            all_pred_labels.extend(results["labels"])

    # Nếu không tìm thấy gì trong tất cả các batch
    if not all_pred_scores:
        return []

    # Gộp kết quả
    final_boxes = torch.cat(all_pred_boxes, dim=0)
    final_scores = torch.cat(all_pred_scores, dim=0)

    # Áp dụng NMS để loại bỏ box trùng lặp giữa các batch
    # (Ví dụ: batch 1 tìm thấy 'cat', batch 2 tìm thấy 'animal' cùng chỗ)
    keep_indices = ops.nms(final_boxes, final_scores, iou_threshold=0.5)

    filtered_results = []
    for idx in keep_indices:
        idx = idx.item()
        filtered_results.append({
            "label": all_pred_labels[idx],
            "score": final_scores[idx].item(),
            "box": final_boxes[idx].tolist() # [xmin, ymin, xmax, ymax]
        })

    return filtered_results

##############################################
# Xử lý tất cả video trong thư mục
##############################################
import time
import os
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CẤU HÌNH ---
OUTPUT_PATH = "../data/MSVD_raw/MSVD_DinoObjects.hdf5"
TIME_LIMIT_SECONDS = 20 * 3600 - 300  # 20 tiếng (trừ hao 5 phút để lưu an toàn)

# --- CHUẨN BỊ ---
video_paths = sorted(Path("../data/MSVD_raw/raw_video/").glob("*.avi"))
keyframe_df = pd.read_csv("../data/MSVD_raw/MSVD_keyframe_counts.csv")
vid_to_keyframe_count = dict(zip(keyframe_df.video_id, keyframe_df.keyframe_count))

# --- 1. KIỂM TRA CHECKPOINT (RESUME) ---
processed_ids = set()
if os.path.exists(OUTPUT_PATH):
    try:
        with h5py.File(OUTPUT_PATH, 'r') as f:
            processed_ids = set(f.keys())
        print(f"🔄 Tìm thấy file checkpoint cũ. Đã xử lý: {len(processed_ids)} videos.")
    except OSError:
        print("⚠️ File HDF5 bị lỗi hoặc không mở được. Sẽ tạo mới (hoặc bạn cần xóa file cũ đi).")

# Lọc ra các video chưa chạy
videos_to_process = [p for p in video_paths if p.stem not in processed_ids]
print(f"▶️ Số video còn lại cần xử lý: {len(videos_to_process)}")

if len(videos_to_process) == 0:
    print("✅ Đã xử lý xong toàn bộ video!")
else:
    # --- 2. VÒNG LẶP XỬ LÝ ---
    start_time = time.time()
    stop_flag = False

    # Dùng tqdm trên danh sách đã lọc
    pbar = tqdm(videos_to_process, desc="Processing")

    for vp in pbar:
        # Kiểm tra thời gian
        elapsed_time = time.time() - start_time
        if elapsed_time > TIME_LIMIT_SECONDS:
            print(f"\n⏰ Đã đạt giới hạn thời gian ({elapsed_time/3600:.2f} giờ). Dừng an toàn để lưu checkpoint.")
            stop_flag = True
            break

        video_id = vp.stem

        # --- A. Trích xuất Keyframes ---
        frames = extract_keyframes(vp)

        # Kiểm tra số lượng (Optional)
        expected_count = vid_to_keyframe_count.get(video_id, 0)
        if len(frames) != expected_count:
             # Chỉ in warning, không dừng
             pass

        # --- B. Chạy Grounding DINO ---
        video_objects = []
        for frame_img in frames:
            # Gọi hàm detect (đã định nghĩa ở cell trước)
            detected_objs = detect_objects_in_frame(
                frame_img,
                entities_list, # Đảm bảo biến này đã load từ file txt
                processor,
                model,
                device
            )
            video_objects.append(detected_objs)

        # --- C. Lưu ngay lập tức (Append Mode) ---
        # Chuyển đổi sang JSON string để lưu vào HDF5
        dt = h5py.string_dtype(encoding='utf-8')
        frame_json_strings = [json.dumps(objs) for objs in video_objects]

        try:
            # Mở file mode 'a' (append), ghi xong đóng ngay để an toàn dữ liệu
            with h5py.File(OUTPUT_PATH, "a") as hf:
                if video_id in hf:
                    del hf[video_id] # Xóa nếu key bị trùng (trường hợp lỗi ghi dở dang)

                hf.create_dataset(
                    video_id,
                    data=np.array(frame_json_strings, dtype=object),
                    dtype=dt
                )
        except Exception as e:
            print(f"\n❌ Lỗi khi lưu video {video_id}: {e}")
            continue # Bỏ qua video lỗi, chạy tiếp

    if stop_flag:
        print(f"\n⏸️ Đã tạm dừng. Lần sau chạy lại cell này, code sẽ tự động tiếp tục từ video tiếp theo.")
    else:
        print(f"\n✅ Hoàn thành batch hiện tại!")