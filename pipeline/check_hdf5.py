"""
HDF5 Utility Functions for Object Detection Results
====================================================
Hỗ trợ cả format cũ (DINO) và format mới (Grounded SAM 2 với tracking).
"""

import json
import h5py
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional


def list_video_ids(h5_path) -> List[str]:
    """Liệt kê tất cả video IDs trong HDF5 file."""
    with h5py.File(h5_path, "r") as hf:
        return list(hf.keys())


def load_video_frames(video_id: str, h5_path) -> List[List[Dict]]:
    """
    Trả về list các frame objects cho video_id.
    
    Returns:
        List[List[Dict]]: Mỗi phần tử là list dict của detections trong 1 frame.
        
    Format detection dict:
        - label: str - Tên object
        - score: float - Confidence score
        - box: [x1, y1, x2, y2] - Bounding box coordinates
        - mask_rle: dict (optional) - RLE encoded mask từ SAM 2
        - object_id: int (optional) - Tracking ID xuyên suốt video
        - frame_idx: int (optional) - Index của frame
    """
    with h5py.File(h5_path, "r") as hf:
        if video_id not in hf:
            raise KeyError(f"{video_id} not found in HDF5.")
        frame_json_strings = hf[video_id][:]
    
    # Giải mã bytes -> str rồi parse JSON
    return [json.loads(s.decode("utf-8") if isinstance(s, bytes) else s) for s in frame_json_strings]


def load_video_with_tracking(video_id: str, h5_path) -> Dict[str, Any]:
    """
    Load video detections và organize theo object_id (tracking).
    
    Returns:
        Dict với keys:
        - 'frames': List[List[Dict]] - Detections per frame
        - 'objects': Dict[int, List[Dict]] - Detections grouped by object_id
        - 'num_frames': int
        - 'num_unique_objects': int
    """
    frames = load_video_frames(video_id, h5_path)
    
    # Group by object_id
    objects_by_id = {}
    for frame_idx, frame_dets in enumerate(frames):
        for det in frame_dets:
            obj_id = det.get("object_id", None)
            if obj_id is not None:
                if obj_id not in objects_by_id:
                    objects_by_id[obj_id] = []
                det_copy = det.copy()
                det_copy["frame_idx"] = frame_idx
                objects_by_id[obj_id].append(det_copy)
    
    return {
        "frames": frames,
        "objects": objects_by_id,
        "num_frames": len(frames),
        "num_unique_objects": len(objects_by_id),
    }


def get_frame_summary(video_id: str, h5_path, frame_idx: int = 0) -> Dict[str, Any]:
    """Lấy thông tin tóm tắt của một frame cụ thể."""
    frames = load_video_frames(video_id, h5_path)
    
    if frame_idx >= len(frames):
        raise IndexError(f"Frame {frame_idx} out of range (total: {len(frames)})")
    
    frame_dets = frames[frame_idx]
    
    return {
        "frame_idx": frame_idx,
        "num_objects": len(frame_dets),
        "labels": [d["label"] for d in frame_dets],
        "has_masks": any("mask_rle" in d for d in frame_dets),
        "has_tracking": any("object_id" in d for d in frame_dets),
        "detections": frame_dets,
    }


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode RLE (Run-Length Encoding) thành binary mask.
    
    Args:
        rle: Dict với 'counts' (list of ints) và 'size' ([height, width])
    
    Returns:
        np.ndarray: Binary mask với shape (height, width)
    """
    if rle is None:
        return None
    
    shape = rle["size"]
    counts = rle["counts"]
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    position = 0
    
    for i, count in enumerate(counts):
        if i % 2 == 1:  # Odd indices are 1s
            mask[position:position + count] = 1
        position += count
    
    return mask.reshape(shape)


def get_video_statistics(h5_path) -> Dict[str, Any]:
    """Thống kê tổng quan về HDF5 file."""
    video_ids = list_video_ids(h5_path)
    
    total_frames = 0
    total_detections = 0
    label_counts = {}
    has_masks = False
    has_tracking = False
    
    for vid in video_ids:
        try:
            frames = load_video_frames(vid, h5_path)
            total_frames += len(frames)
            
            for frame_dets in frames:
                total_detections += len(frame_dets)
                for det in frame_dets:
                    label = det.get("label", "unknown")
                    label_counts[label] = label_counts.get(label, 0) + 1
                    
                    if "mask_rle" in det:
                        has_masks = True
                    if "object_id" in det:
                        has_tracking = True
        except Exception as e:
            print(f"Warning: Error reading {vid}: {e}")
    
    # Sort labels by count
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "num_videos": len(video_ids),
        "total_frames": total_frames,
        "total_detections": total_detections,
        "avg_detections_per_frame": total_detections / max(total_frames, 1),
        "unique_labels": len(label_counts),
        "top_10_labels": sorted_labels[:10],
        "has_masks": has_masks,
        "has_tracking": has_tracking,
    }


def convert_to_legacy_format(video_id: str, h5_path_new, h5_path_legacy):
    """
    Convert từ format mới (Grounded SAM 2) sang format cũ (compatible với predict_rotate_relations.py).
    Loại bỏ mask_rle và object_id, chỉ giữ label, score, box.
    """
    frames = load_video_frames(video_id, h5_path_new)
    
    legacy_frames = []
    for frame_dets in frames:
        legacy_dets = []
        for det in frame_dets:
            legacy_dets.append({
                "label": det["label"],
                "score": det["score"],
                "box": det["box"],
            })
        legacy_frames.append(legacy_dets)
    
    # Save to legacy HDF5
    dt = h5py.string_dtype(encoding="utf-8")
    frame_json_strings = [json.dumps(objs) for objs in legacy_frames]
    
    with h5py.File(h5_path_legacy, "a") as hf:
        if video_id in hf:
            del hf[video_id]
        hf.create_dataset(
            video_id,
            data=np.array(frame_json_strings, dtype=object),
            dtype=dt,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check HDF5 file for video detection data.")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to the HDF5 file to analyze.")
    parser.add_argument("--video_id", type=str, default=None, help="Specific video ID to inspect.")
    parser.add_argument("--stats", action="store_true", help="Show overall statistics.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to show (default: 0).")
    args = parser.parse_args()
    
    hdf5_path = Path(args.hdf5_path)
    print(f"Reading: {hdf5_path.resolve()}")
    
    if args.stats:
        print("\n" + "=" * 60)
        print("OVERALL STATISTICS")
        print("=" * 60)
        stats = get_video_statistics(hdf5_path)
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        video_ids = list_video_ids(hdf5_path)
        print(f"Found {len(video_ids)} videos. First 5 IDs: {video_ids[:5]}")

        if args.video_id:
            vid = args.video_id
        elif video_ids:
            vid = video_ids[0]
        else:
            print("No videos found.")
            exit(1)
        
        print(f"\n" + "=" * 60)
        print(f"VIDEO: {vid}")
        print("=" * 60)
        
        frames = load_video_frames(vid, hdf5_path)
        print(f"Total keyframes: {len(frames)}")
        
        # Show specific frame
        if args.frame < len(frames):
            summary = get_frame_summary(vid, hdf5_path, args.frame)
            print(f"\nFrame {args.frame} summary:")
            print(f"  Objects: {summary['num_objects']}")
            print(f"  Labels: {summary['labels']}")
            print(f"  Has masks: {summary['has_masks']}")
            print(f"  Has tracking: {summary['has_tracking']}")
            print(f"\nDetections:")
            for det in summary['detections'][:5]:  # Show first 5
                print(f"  - {det['label']}: {det['score']:.3f} @ {det['box']}")
        
        # Check tracking info
        tracking_info = load_video_with_tracking(vid, hdf5_path)
        if tracking_info["num_unique_objects"] > 0:
            print(f"\nTracking info:")
            print(f"  Unique tracked objects: {tracking_info['num_unique_objects']}")
            for obj_id, appearances in list(tracking_info['objects'].items())[:3]:
                print(f"  Object {obj_id}: appears in {len(appearances)} frames")