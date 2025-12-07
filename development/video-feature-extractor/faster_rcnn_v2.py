import numpy as np
import pandas as pd
import h5py
import av
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as TF
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


# ==================== Setup Model ====================
def load_model_and_transform(device):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    transform = weights.transforms()  # inference transforms
    return model, transform


# ==================== Keyframe Extraction ====================
def extract_keyframes(video_path: str) -> List[Image.Image]:
    """
    Trả về list các PIL.Image tương ứng keyframes.
    """
    keyframes = []
    container = av.open(video_path)
    for frame in container.decode(video=0):
        if frame.key_frame:
            img = frame.to_image().convert("RGB")
            keyframes.append(img)
    container.close()
    return keyframes


# ==================== Image Resizing & Tensor Utils ====================
def resize_keep_aspect_and_tensor(img: Image.Image,
                                  transform,
                                  short_side: int = 800,
                                  max_size: int = 1333) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[float, float]]:
    """
    Resize PIL image so that shorter side == short_side (capped by max_size on long side),
    keep aspect ratio. Return:
      - tensor [C,H,W] normalized (float, not batched)
      - resized size (H, W) as ints
      - scale factors (scale_h, scale_w) = resized_dim / original_dim
    """
    orig_w, orig_h = img.size  # PIL: (W,H)
    # compute scale
    min_orig = min(orig_h, orig_w)
    max_orig = max(orig_h, orig_w)
    scale = float(short_side) / float(min_orig)
    # if after scaling, longer side would exceed max_size, reduce scale
    if round(scale * max_orig) > max_size:
        scale = float(max_size) / float(max_orig)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    resized = F.resize(img, (new_h, new_w)) # PIL Image of size (H, W)
    tensor = transform(resized)             # Tensor of shape [C, H, W], values [0,1]
    scale_h = new_h / orig_h
    scale_w = new_w / orig_w
    return tensor, (new_h, new_w), (scale_h, scale_w)


def pad_batch_tensors(tensors: List[torch.Tensor], device) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Given list of [C,H_i,W_i] tensors, pad them to (C, H_max, W_max) on bottom/right and stack into [B,C,H_max,W_max].
    Returns stacked tensor (on same device as inputs) and list of (H_i,W_i).
    """
    device_local = tensors[0].device if tensors else device
    C = tensors[0].shape[0]
    heights = [t.shape[1] for t in tensors]
    widths = [t.shape[2] for t in tensors]
    H_max = max(heights)
    W_max = max(widths)
    padded = []
    shapes = []
    for t in tensors:
        h, w = t.shape[1], t.shape[2]
        pad_bottom = H_max - h
        pad_right  = W_max - w
        # F.pad takes (left, top, right, bottom) when using torch.nn.functional.pad for 3d? For 3D tensor: pad = (pad_left, pad_right, pad_top, pad_bottom)
        # But for 3D [C,H,W], pad expects (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
        t_p = TF.pad(t, (0, pad_right, 0, pad_bottom))  # pad right then bottom
        padded.append(t_p)
        shapes.append((h, w))
    batch = torch.stack(padded, dim=0).to(device_local)  # [B, C, H_max, W_max]
    return batch, shapes


# ==================== Main Extraction Pipeline ====================
@torch.no_grad()
def extract_embeddings_from_keyframes(keyframes: List[Image.Image],
                                      model,
                                      transform,
                                      device,
                                      batch_size: int = 8,
                                      short_side: int = 800,
                                      max_size: int = 1333,
                                      score_thresh: float = 0.5,
                                      max_objects_per_frame: int = 50,
                                      aggregate: str = "mean"):
    """
    Input:
      - keyframes: list of PIL.Image (RGB)
      - model: torchvision Faster R-CNN model (pretrained), .eval() and on device
    Returns:
      - results: list with one element per frame:
          {
            'frame_index': int,
            'objects': [ { 'box': np.array([x1,y1,x2,y2]) (in original image pixels),
                           'score': float,
                           'label': int or None,
                           'embedding': np.array(...) } , ... ],
            'image_embedding': np.array(...)
          }
    Notes:
      - We process keyframes in batches. For each batch:
         * resize each image (keep aspect ratio),
         * run model(images_list) to get detections (boxes in resized coordinates),
         * create a padded batch tensor and run model.backbone once,
         * pool features with box_roi_pool and run box_head once for all boxes in batch.
      - Boxes are converted from resized coords back to original image pixel coords.
    """
    model.eval()
    results = []

    # Precompute resized tensors and scales for all frames
    prepared = []
    for i, img in enumerate(keyframes):
        t, resized_hw, (scale_h, scale_w) = resize_keep_aspect_and_tensor(
            img, transform=transform, short_side=short_side, max_size=max_size)
        prepared.append({
            'orig_size': (img.height, img.width),          # H_orig, W_orig
            'resized_size': resized_hw,                    # H_resized, W_resized
            'scale': (scale_h, scale_w),
            'tensor': t
        })

    # process in batches
    N = len(prepared)
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        batch_items = prepared[start:end]

        # 1) detections using model on list of per-image tensors (no padding here)
        # list of [C,H_i,W_i] on device
        images_list = [item['tensor'].to(device) for item in batch_items]
        # list of dicts per image (boxes in resized image coords)
        outputs = model(images_list)

        # 2) prepare padded batch for backbone (to run backbone once)
        # Use the same tensors but pad to max H,W
        batch_tensor, shapes = pad_batch_tensors(images_list, device)  # [B,C,Hmax,Wmax] on device

        # 3) run backbone once on batch tensor -> OrderedDict of feature maps each [B,C_i,H_i,W_i]
        features = model.backbone(batch_tensor)

        # 4) build boxes_for_pool: list of tensors (boxes per image) in same device dtype
        boxes_for_pool = []
        scores_for_pool = []
        labels_for_pool = []
        # number of kept boxes per image (after score thresh & top-k)
        valid_counts = []
        for out in outputs:
            boxes = out['boxes'].to(device)
            scores = out['scores'].to(device)
            labels = out.get('labels', None)
            if labels is not None:
                labels = labels.to(device)

            # filter by score
            keep_mask = scores >= score_thresh
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            if labels is not None:
                labels = labels[keep_mask]

            if boxes.numel() == 0:
                # empty -> append empty tensor (shape [0,4])
                boxes_for_pool.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                scores_for_pool.append(torch.zeros((0,), dtype=torch.float32, device=device))
                labels_for_pool.append(torch.zeros((0,), dtype=torch.int64, device=device) if labels is None else labels)
                valid_counts.append(0)
                continue

            # top-k
            scores_sorted, idx = scores.sort(descending=True)
            idx = idx[:max_objects_per_frame]
            boxes = boxes[idx]
            scores = scores_sorted[:max_objects_per_frame]
            if labels is not None:
                labels = labels[idx]

            boxes_for_pool.append(boxes)
            scores_for_pool.append(scores)
            labels_for_pool.append(labels if labels is not None else None)
            valid_counts.append(boxes.shape[0])

        # 5) compute image_shapes list used by roi_pool: use resized sizes (H_resized, W_resized)
        image_shapes = [item['resized_size'] for item in batch_items]  # list of (H,W)

        # 6) pooled features for all boxes in this batch
        # box_roi_pool expects boxes_for_pool (list length = batch_size)
        pooled = model.roi_heads.box_roi_pool(features, boxes_for_pool, image_shapes)
        # pooled -> [total_boxes, C, 7, 7] (7x7 default)

        if pooled.shape[0] == 0:
            # no boxes in entire batch
            # produce empty results accordingly
            for idx_in_batch, item in enumerate(batch_items):
                results.append({
                    'frame_index': start + idx_in_batch,
                    'objects': [],
                    # fallback dim
                    'image_embedding': np.zeros(1024, dtype=np.float32)
                })
            continue

        # 7) box_head -> feature vector per box
        box_features = model.roi_heads.box_head(pooled)  # [total_boxes, feat_dim]
        box_features = TF.normalize(box_features, p=2, dim=1)

        # 8) Now split box_features back into per-image groups using valid_counts
        per_image_embeddings = []
        ptr = 0
        for count in valid_counts:
            if count == 0:
                per_image_embeddings.append(torch.zeros((0, box_features.shape[1]), device=device))
            else:
                per_image_embeddings.append(box_features[ptr:ptr+count])
            ptr += count

        # 9) For each image in batch, prepare objects list and image embedding
        for i_in_batch, (item, out, boxes_resized, scores_resized, labels_resized, emb_t) in enumerate(
                zip(batch_items, outputs, boxes_for_pool, scores_for_pool, labels_for_pool, per_image_embeddings)):
            frame_idx = start + i_in_batch
            objs = []
            if emb_t.shape[0] == 0:
                image_embedding = np.zeros(box_features.shape[1], dtype=np.float32)
            else:
                # normalize boxes back to original image pixel coords
                H_orig, W_orig = item['orig_size']
                scale_h, scale_w = item['scale']
                # emb_t: [K, feat_dim]; boxes_resized: [K,4] in resized pixel coords (x1,y1,x2,y2)
                # convert boxes to original: orig_x = resized_x / scale_w; orig_y = resized_y / scale_h
                if boxes_resized.numel() == 0:
                    boxes_orig = torch.zeros((0, 4), device=device)
                else:
                    boxes_orig = boxes_resized.clone()
                    boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]] / scale_w
                    boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]] / scale_h

                """
                # concat features + normalized coords (by original W/H) -> embedding dim = feat_dim + 4
                coords_norm = boxes_orig.clone()
                coords_norm[:, [0, 2]] = coords_norm[:, [0, 2]] / float(W_orig)  # x by width
                coords_norm[:, [1, 3]] = coords_norm[:, [1, 3]] / float(H_orig)  # y by height

                emb_full = torch.cat([emb_t, coords_norm.to(emb_t.dtype)], dim=1)  # [K, feat_dim+4]
                """;
                
                # ! Only use features as embedding
                emb_full = torch.tensor(emb_t, device=device)  # [K, feat_dim]
                
                # prepare object dicts
                for k in range(emb_full.shape[0]):
                    emb_np = emb_full[k].cpu().numpy()
                    box_np = boxes_orig[k].cpu().numpy()
                    score_val = float(scores_resized[k].cpu().item()) if scores_resized.numel() else 0.0
                    label_val = int(labels_resized[k].cpu().item()) if (labels_resized is not None and labels_resized.numel()) else None
                    objs.append({
                        'box': box_np,   # in original image pixel coords
                        'score': score_val,
                        'label': label_val,
                        'embedding': emb_np
                    })

                # image-level aggregation
                if aggregate == "mean":
                    image_embedding = emb_full.mean(dim=0).cpu().numpy()
                elif aggregate == "attention":
                    # use detection scores as attention weights
                    w = torch.softmax(scores_resized.to(emb_t.dtype), dim=0).unsqueeze(1)
                    image_embedding = (emb_full * w).sum(dim=0).cpu().numpy()
                else:
                    image_embedding = emb_full.mean(dim=0).cpu().numpy()

            results.append({
                'frame_index': frame_idx,
                'objects': objs,
                'image_embedding': image_embedding
            })

    return results


def run() -> None:
    # load model + transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_model_and_transform(device)
    
    # Tạo một dictionary ánh xạ từ video_id đến số lượng khung hình chính (keyframes)
    keyframe_df = pd.read_csv("./MSVD_keyframe_counts.csv")
    vid_to_keyframe_count = {video_id: keyframe_count
                             for video_id, keyframe_count in zip(keyframe_df["video_id"], keyframe_df["keyframe_count"])}

    # Lấy danh sách tất cả các video trong thư mục raw_video
    video_paths = sorted(Path("./raw_video/").glob("*.avi"))
    assert len(video_paths) == 1970

    # open HDF5
    feature_file_path = "./MSVD_FasterRCNNv2.hdf5"
    print(f"Extracting features and saving to {feature_file_path}...")
    with h5py.File(feature_file_path, "w") as hf:
        for vp in tqdm(video_paths, desc="Videos"):
            # get keyframe count for this video
            keyframe_count = vid_to_keyframe_count[vp.stem]
            # extract keyframes
            keyframes = extract_keyframes(str(vp))
            assert len(keyframes) == keyframe_count, f"Keyframe count mismatch for {vp.stem}: expected {keyframe_count}, got {len(keyframes)}"

            # extract features (returns numpy (N, hidden_dim))
            results = extract_embeddings_from_keyframes(keyframes, model,
                                                        transform, device,
                                                        batch_size=16,
                                                        short_side=800,
                                                        max_size=1333,
                                                        score_thresh=0.5,
                                                        max_objects_per_frame=6,
                                                        aggregate="mean")
            img_emb_list = [result['image_embedding'] for result in results]
            feats = np.vstack(img_emb_list).astype(np.float32)
            assert feats.shape[0] == keyframe_count, f"Feature count mismatch for {vp.stem}: expected {keyframe_count}, got {feats.shape[0]}"
            assert len(feats.shape) == 2, f"Feature shape invalid for {vp.stem}: got {feats.shape}"
            assert feats.shape[1] == 1024, f"Feature dimension mismatch for {vp.stem}: expected 1024, got {feats.shape[1]}"

            # Create dataset
            hf.create_dataset(vp.stem, data=feats)

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
            assert f[key].shape[1] == 1024

    print("Done!")


if __name__ == "__main__":
    run()
