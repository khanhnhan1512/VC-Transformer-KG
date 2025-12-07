import numpy as np
import pandas as pd
import h5py
import av
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import Blip2Processor, Blip2ForConditionalGeneration


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


def extract_features_for_frames(frames_pil, processor, vision_model, device, batch_size):
    """
    frames_pil: list of PIL.Image
    processor: BlipProcessor
    vision_model: model.vision_model
    returns: numpy array shape (len(frames_pil), hidden_dim), dtype float32
    """
    all_feats = []
    vision_model.to(device)
    vision_model.eval()

    with torch.no_grad():
        for i in range(0, len(frames_pil), batch_size):
            batch_imgs = frames_pil[i:i+batch_size]

            # processor will resize/normalize according to model's default transforms
            inputs = processor(images=batch_imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            # inference
            outputs = vision_model(pixel_values=pixel_values)

            # BLIP vision_model returns last_hidden_state and pooler_output (depending on model)
            pooled = outputs.pooler_output  # (B, hidden_dim)
            pooled = pooled.cpu().numpy().astype(np.float32)

            all_feats.append(pooled)

    if len(all_feats) == 0:
        raise ValueError("No features extracted")

    feats = np.vstack(all_feats)  # shape (N_frames, hidden_dim)
    return feats


# Tạo một dictionary ánh xạ từ video_id đến số lượng khung hình chính (keyframes)
keyframe_df = pd.read_csv("./MSVD_keyframe_counts.csv")
vid_to_keyframe_count = {video_id: keyframe_count
                         for video_id, keyframe_count in zip(keyframe_df["video_id"], keyframe_df["keyframe_count"])}

cache_dir = "/media02/lnthanh01/vmphat/cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir).to(device)
vision_model = model.vision_model  # encoder for images


video_paths = sorted(Path("./raw_video/").glob("*.avi"))
assert len(video_paths) == 1970

# open HDF5
feature_file_path = "./MSVD_Blip2ClsKF.hdf5"
with h5py.File(feature_file_path, "w") as hf:
    for vp in tqdm(video_paths, desc="Videos"):
        # get keyframe count for this video
        keyframe_count = vid_to_keyframe_count[vp.stem]
        # extract keyframes
        keyframes = extract_keyframes(str(vp))
        assert len(keyframes) == keyframe_count, f"Keyframe count mismatch for {vp.stem}: expected {keyframe_count}, got {len(keyframes)}"

        # extract features (returns numpy (N, hidden_dim))
        feats = extract_features_for_frames(keyframes, processor, vision_model, device, batch_size=8)
        assert feats.shape[0] == keyframe_count, f"Feature count mismatch for {vp.stem}: expected {keyframe_count}, got {feats.shape[0]}"
        assert len(feats.shape) == 2, f"Feature shape invalid for {vp.stem}: got {feats.shape}"

        # Create dataset
        hf.create_dataset(vp.stem, data=feats)
print("Done.")


# Verification step: đọc lại và kiểm tra
print(f"Verifying saved HDF5 file...")
with h5py.File(feature_file_path, "r") as f:
    # liệt kê các key
    keys = list(f.keys())
    print("#keys:", len(keys))
    assert len(keys) == 1970

    # lấy một key (ví dụ key đầu tiên)
    key = keys[0]

    # dataset object (không đọc toàn bộ vào nhớ)
    dset = f[key]
    print("shape:", dset.shape, "dtype:", dset.dtype)

    # đọc toàn bộ vào numpy array
    arr = dset[:]          # hoặc np.array(dset)
    print("arr shape:", arr.shape)

    for key in tqdm(keys):
        assert len(f[key].shape) == 2
        assert f[key].shape[0] == vid_to_keyframe_count[key]
print("Done!")
