#!/usr/bin/env python3
"""
extract_gop_frames.py — Extract all decoded frames from a video, organized by GOP.

Creates one folder per GOP (GOP001, GOP002, ...) in the current directory.
Each folder contains lossless PNG frames named in display order:
    000001_I.png, 000002_B.png, 000003_B.png, 000004_P.png, ...

Requirements:
    pip install opencv-python
    ffprobe (FFmpeg) must be in PATH

Usage:
    python extract_gop_frames.py path/to/video.mp4
"""

import subprocess
import json
import sys
import os
import argparse

try:
    import cv2
except ImportError:
    sys.exit("Error: pip install opencv-python")


# PNG compression level (0-9). All levels are LOSSLESS — only affects
# file size and write speed. 0 = no compression, 9 = max compression.
PNG_COMPRESSION = 3


def analyze_gops(video_path):
    """Use ffprobe to get frame types in display order, grouped into GOPs."""
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "frame=pict_type,key_frame,best_effort_timestamp_time",
        "-of", "json", str(video_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        sys.exit("ffprobe not found. Install FFmpeg and add it to PATH.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffprobe failed: {e.stderr}")

    raw = json.loads(r.stdout).get("frames", [])
    if not raw:
        sys.exit("No video frames found.")

    # Parse PTS for display-order sorting
    for i, f in enumerate(raw):
        pts = f.get("best_effort_timestamp_time", "N/A")
        f["_pts"] = float(pts) if pts not in ("N/A", None, "") else i / 30.0

    raw.sort(key=lambda f: f["_pts"])

    # Group into GOPs (split at each keyframe)
    gops, cur = [], []
    for di, f in enumerate(raw):
        is_key = int(f.get("key_frame", 0)) == 1 or f["pict_type"] == "I"
        if is_key and cur:
            gops.append(cur)
            cur = []
        cur.append({"type": f["pict_type"], "di": di})
    if cur:
        gops.append(cur)

    return gops


def extract_and_save(video_path, gops):
    """Read video sequentially and save each frame to its GOP folder."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    # Build lookup: display_index -> (gop_idx, position_in_gop, frame_type)
    lookup = {}
    for gi, gop in enumerate(gops):
        for fi, finfo in enumerate(gop):
            lookup[finfo["di"]] = (gi, fi, finfo["type"])

    # Create GOP folders
    for gi in range(len(gops)):
        os.makedirs(f"GOP{gi + 1:03d}", exist_ok=True)

    # Sequential read — ensures correct display-order extraction
    total = len(lookup)
    saved = 0
    idx = 0
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]

    while saved < total:
        ok = cap.grab()
        if not ok:
            break
        if idx in lookup:
            ok, frame = cap.retrieve()
            if ok:
                gi, fi, ftype = lookup[idx]
                path = os.path.join(f"GOP{gi + 1:03d}", f"{fi + 1:06d}_{ftype}.png")
                cv2.imwrite(path, frame, png_params)
                saved += 1
                if saved % 100 == 0 or saved == total:
                    print(f"  [{saved}/{total}]")
        idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frames organized by GOP structure.")
    parser.add_argument("video", help="Path to input video file")
    args = parser.parse_args()

    # 1. Analyze
    print(f"Analyzing: {args.video}")
    gops = analyze_gops(args.video)

    total_frames = sum(len(g) for g in gops)
    print(f"\nFound {len(gops)} GOPs, {total_frames} total frames:\n")
    print(f"  {'GOP':<8} {'Frames':>6}   {'I':>3} {'P':>3} {'B':>3}   Structure")
    print(f"  {'---':<8} {'------':>6}   {'---':>3} {'---':>3} {'---':>3}   ---------")
    for i, g in enumerate(gops):
        types = "".join(f["type"] for f in g)
        n_i = sum(1 for f in g if f["type"] == "I")
        n_p = sum(1 for f in g if f["type"] == "P")
        n_b = sum(1 for f in g if f["type"] == "B")
        struct = types[:45] + ("..." if len(types) > 45 else "")
        print(f"  GOP{i+1:03d}   {len(g):>6}   {n_i:>3} {n_p:>3} {n_b:>3}   {struct}")

    # 2. Extract
    print(f"\nExtracting to current directory...")
    saved = extract_and_save(args.video, gops)
    print(f"\nDone. {saved}/{total_frames} frames saved to {len(gops)} folders.")


if __name__ == "__main__":
    main()