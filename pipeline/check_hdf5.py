import json
import h5py
from pathlib import Path
import argparse

def list_video_ids(h5_path):
    with h5py.File(h5_path, "r") as hf:
        return list(hf.keys())

def load_video_frames(video_id, h5_path):
    """Trả về list các frame objects cho video_id (mỗi phần tử là list dict)."""
    with h5py.File(h5_path, "r") as hf:
        if video_id not in hf:
            raise KeyError(f"{video_id} not found in HDF5.")
        frame_json_strings = hf[video_id][:]
    # Giải mã bytes -> str rồi parse JSON
    return [json.loads(s.decode("utf-8") if isinstance(s, bytes) else s) for s in frame_json_strings]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check HDF5 file for video data.")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to the HDF5 file to analyze.")
    args = parser.parse_args()
    
    hdf5_path = Path(args.hdf5_path)
    print(f"Reading: {hdf5_path.resolve()}")
    video_ids = list_video_ids(hdf5_path)
    print(f"Found {len(video_ids)} videos. First 5 IDs: {video_ids[:5]}")

    if video_ids:
        vid = video_ids[0]
        frames = load_video_frames(vid, hdf5_path)
        print(f"\nVideo '{vid}' has {len(frames)} keyframes.")
        # Mẫu in 1-2 frame đầu
        for i, objs in enumerate(frames[:2]):
            print(f"Frame {i}: {objs}")