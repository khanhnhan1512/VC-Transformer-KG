# coding=utf-8
"""Extract GOP-level data (I-frames + motion vectors) from a video using PyAV.

Video = chuỗi GOP; mỗi GOP mở đầu bằng 1 I-frame (keyframe, do bước tiền xử lý
video quyết định qua GOP size) và theo sau là các P/B-frame. P/B-frame trong
video nén được lưu dưới dạng motion vectors + residuals — ta trích MV làm tín
hiệu chuyển động cho mỗi GOP mà không cần decode ảnh hay tính optical flow.
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


# Giá trị AVFrameSideDataType.AV_FRAME_DATA_MOTION_VECTORS trong FFmpeg
# (ABI ổn định — chỉ append giá trị mới, không đổi giá trị cũ)
_MV_SIDE_DATA_TYPE_INT = 8


def _get_motion_vectors(frame):
    """Lấy MV side data của 1 frame đã decode (None nếu không có).

    Tương thích nhiều phiên bản PyAV: sd.type có thể là enum (có .name),
    int thô, hoặc nhận diện qua class wrapper MotionVectors.
    """
    for sd in frame.side_data:
        type_name = getattr(sd.type, 'name', None)
        if (type_name == 'MOTION_VECTORS'
                or sd.__class__.__name__ == 'MotionVectors'
                or (type_name is None and int(sd.type) == _MV_SIDE_DATA_TYPE_INT)):
            return sd.to_ndarray()
    return None


def _rasterize_mvs(mv_arr, frame_width, frame_height, grid_size):
    """Gom các motion vector của 1 frame về lưới (2, grid, grid).

    Mỗi ô lưới = displacement trung bình (dx, dy) của các MV có đích rơi vào ô,
    chuẩn hóa theo kích thước frame (giá trị ~[-1, 1], bất biến với độ phân giải).
    """
    grid = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    if mv_arr is None or len(mv_arr) == 0:
        return grid

    scale = mv_arr['motion_scale'].astype(np.float32)
    scale[scale == 0] = 1.0
    dx = (mv_arr['motion_x'] / scale) / float(frame_width)
    dy = (mv_arr['motion_y'] / scale) / float(frame_height)

    # B-frame có MV tham chiếu tương lai (source > 0): hướng hình học ngược với
    # chuyển động thật -> đảo dấu để mọi MV cùng quy ước "quá khứ -> hiện tại",
    # tránh trung bình bị triệt tiêu khi trộn MV backward/forward trong cùng GOP
    backward = mv_arr['source'] > 0
    dx = np.where(backward, -dx, dx)
    dy = np.where(backward, -dy, dy)

    gx = np.clip(mv_arr['dst_x'] * grid_size // frame_width, 0, grid_size - 1).astype(int)
    gy = np.clip(mv_arr['dst_y'] * grid_size // frame_height, 0, grid_size - 1).astype(int)

    count = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(grid[0], (gy, gx), dx.astype(np.float32))
    np.add.at(grid[1], (gy, gx), dy.astype(np.float32))
    np.add.at(count, (gy, gx), 1.0)
    count[count == 0] = 1.0
    return grid / count


def extract_gop_data(video_path: str, threshold: int, size: int = 224,
                     grid_size: int = 16):
    """Decode video MỘT lượt, thu dữ liệu mức GOP:

    - I-frame (ảnh RGB) của mỗi GOP -> token appearance
    - Motion vector map tổng hợp của các P/B-frame trong GOP -> token motion
    - Timestamp (giây) của I-frame -> positional encoding theo thời gian thật

    Chuẩn hóa số GOP về `threshold` giống extract_keyframes: thiếu -> zero-pad
    (kèm num_real để xây mask), thừa -> uniform sampling trên danh sách GOP.

    Returns:
        frames:      (threshold, size, size, 3) uint8
        motion_maps: (threshold, 2, grid_size, grid_size) float32
        timestamps:  (threshold,) float32 — giây; 0 tại vị trí pad
        num_real:    số GOP thật, trong [1, threshold]
    """
    keyframes, gop_grids, gop_counts, timestamps = [], [], [], []

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.codec_context.options = {'flags2': '+export_mvs'}
        for frame in container.decode(stream):
            if frame.key_frame:
                arr = frame.reformat(width=size, height=size, format='rgb24').to_ndarray()
                keyframes.append(arr)
                gop_grids.append(np.zeros((2, grid_size, grid_size), dtype=np.float32))
                gop_counts.append(0)
                timestamps.append(float(frame.time) if frame.time is not None else 0.0)
            else:
                if not keyframes:
                    continue  # P/B-frame lạc trước I-frame đầu tiên (hiếm)
                mv_arr = _get_motion_vectors(frame)
                if mv_arr is not None and len(mv_arr) > 0:
                    gop_grids[-1] += _rasterize_mvs(mv_arr, frame.width, frame.height, grid_size)
                    gop_counts[-1] += 1

    num_kf = len(keyframes)
    if num_kf == 0:
        raise ValueError(f"[extract_gop_data] No keyframe decoded from video: {video_path}")

    # MV map của GOP = trung bình các frame có MV trong GOP đó
    for i in range(num_kf):
        if gop_counts[i] > 0:
            gop_grids[i] /= gop_counts[i]

    if num_kf > threshold:
        sampled_idxs = np.linspace(0, num_kf - 1, threshold, dtype=int)
        keyframes = [keyframes[i] for i in sampled_idxs]
        gop_grids = [gop_grids[i] for i in sampled_idxs]
        timestamps = [timestamps[i] for i in sampled_idxs]
        num_real = threshold
    else:
        num_real = num_kf
        n_pad = threshold - num_kf
        keyframes = keyframes + [np.zeros((size, size, 3), dtype=np.uint8)] * n_pad
        gop_grids = gop_grids + [np.zeros((2, grid_size, grid_size), dtype=np.float32)] * n_pad
        timestamps = timestamps + [0.0] * n_pad

    return (np.stack(keyframes, axis=0),
            np.stack(gop_grids, axis=0),
            np.array(timestamps, dtype=np.float32),
            num_real)
