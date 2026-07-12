# coding=utf-8
"""Extract GOP-level motion data (motion vectors + timestamps) using PyAV.

Video = chuỗi GOP; mỗi GOP mở đầu bằng 1 I-frame (keyint=60 từ bước tiền xử
lý) và theo sau là các P/B-frame. P/B-frame trong video nén được lưu dưới
dạng motion vectors + residuals — ta trích MV làm tín hiệu chuyển động cho
mỗi GOP mà không cần dùng pixel đã decode hay chạy mô hình 3D (MViTv2).

Trả về dữ liệu ĐỦ ĐỘ DÀI (một entry cho MỌI GOP, không pad/sample) — giống
quy ước của blip2_extractor.py, để bước chuẩn hóa độ dài được thực hiện
CHUNG với các feature HDF5 trong dataset (đảm bảo alignment 1-1).

Requires: pip install av
"""
import av
import numpy as np

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


def extract_gop_timestamps(video_path: str):
    """Chỉ lấy timestamp (giây) của các I-frame — decode với skip_frame NONKEY
    nên rất nhanh. Dùng cho run ablation không motion nhưng vẫn cần timestamp
    PE (tránh timestamps=0 làm PE suy biến thành hằng số, mất thông tin thứ tự).
    """
    timestamps = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        for frame in container.decode(stream):
            timestamps.append(float(frame.time) if frame.time is not None else 0.0)
    if not timestamps:
        raise ValueError(f"[extract_gop_timestamps] No keyframe decoded: {video_path}")
    return np.array(timestamps, dtype=np.float32)


def extract_gop_motion(video_path: str, grid_size: int = 16):
    """Decode video MỘT lượt, thu motion data cho TẤT CẢ GOP (không pad/sample).

    Returns:
        motion_maps: (num_gop, 2, grid_size, grid_size) float32 — MV map trung
            bình của các P/B-frame trong từng GOP.
        timestamps:  (num_gop,) float32 — giây của I-frame mở đầu mỗi GOP.
        num_gop:     số GOP (= số I-frame) của video.
    """
    gop_grids, gop_counts, timestamps = [], [], []

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.codec_context.options = {'flags2': '+export_mvs'}
        for frame in container.decode(stream):
            if frame.key_frame:
                gop_grids.append(np.zeros((2, grid_size, grid_size), dtype=np.float32))
                gop_counts.append(0)
                timestamps.append(float(frame.time) if frame.time is not None else 0.0)
            else:
                if not gop_grids:
                    continue  # P/B-frame lạc trước I-frame đầu tiên (hiếm)
                mv_arr = _get_motion_vectors(frame)
                if mv_arr is not None and len(mv_arr) > 0:
                    gop_grids[-1] += _rasterize_mvs(mv_arr, frame.width, frame.height, grid_size)
                    gop_counts[-1] += 1

    num_gop = len(gop_grids)
    if num_gop == 0:
        raise ValueError(f"[extract_gop_motion] No keyframe decoded from video: {video_path}")

    # MV map của GOP = trung bình các frame có MV trong GOP đó
    for i in range(num_gop):
        if gop_counts[i] > 0:
            gop_grids[i] /= gop_counts[i]

    return (np.stack(gop_grids, axis=0),
            np.array(timestamps, dtype=np.float32),
            num_gop)
