# coding=utf-8
"""Extract keyframes (I-frames) from a video using PyAV.

The keyframes are decided by the video encoding itself (video preprocessing
step controls the GOP size / keyframe placement), NOT sampled from all frames.
Requires: pip install av
"""
import av
import numpy as np
from typing import Tuple


def extract_keyframes(video_path: str, threshold: int, size: int = 224) -> Tuple[np.ndarray, int]:
    """Decode only the I-frames of a video and normalize their count.

    Args:
        video_path: path to the video file (e.g. .avi)
        threshold: fixed number of keyframes per video
        size: output frame resolution (size x size)

    Returns:
        frames: np.ndarray of shape (threshold, size, size, 3), dtype uint8.
            If the video has fewer keyframes than `threshold`, the tail is
            zero-padded. If it has more, keyframes are uniformly sampled
            down to `threshold`.
        num_real: number of real (non-padded) keyframes, in [1, threshold].
    """
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        # Decode I-frames only: the demuxer/decoder skips all P/B-frames
        stream.codec_context.skip_frame = "NONKEY"
        for frame in container.decode(stream):
            arr = frame.reformat(width=size, height=size, format='rgb24').to_ndarray()
            frames.append(arr)

    num_kf = len(frames)
    if num_kf == 0:
        raise ValueError(f"[extract_keyframes] No keyframe decoded from video: {video_path}")

    if num_kf > threshold:
        # Uniformly sample over the keyframe list (not over all video frames)
        sampled_idxs = np.linspace(0, num_kf - 1, threshold, dtype=int)
        frames = [frames[i] for i in sampled_idxs]
        num_real = threshold
    else:
        num_real = num_kf
        pad = np.zeros((size, size, 3), dtype=np.uint8)
        frames = frames + [pad] * (threshold - num_kf)

    return np.stack(frames, axis=0), num_real
