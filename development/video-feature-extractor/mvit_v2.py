import numpy as np
import pandas as pd
import h5py
import av
import torch
from tqdm import tqdm
from pathlib import Path
from torchvision import io
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import os
import numpy as np
os.environ["TORCH_HOME"] = "/media02/lnthanh01/vmphat/raw_data/cache"


def resample_fixed(frames, chunk_size):
    C, T, H, W = frames.shape
    idxs = np.linspace(0, T-1, chunk_size).astype(int)
    return frames[:, idxs, :, :]


def chunk_frames(inputs, chunk_size, stack_size, step_size):
    """ Chunk input frames into overlapping chunks.

    Args:
        inputs: torch.Tensor of shape (C, T, H, W)
        chunk_size: int, number of frames per chunk after resampling
        stack_size: int, number of frames to stack before resampling
        step_size: int, step size for moving the window

    Returns:
        List of torch.Tensor, each of shape (C, chunk_size, H, W)
    """
    C, T, H, W = inputs.shape
    chunked_inputs = []

    for i in range(0, T, step_size):
        start_idx = i
        end_idx = min(i + stack_size, T)
        num_frames = end_idx - start_idx
        # print(f"Chunk {i}, from {start_idx} to {end_idx} => num_frames={num_frames}")

        if num_frames < chunk_size:
            break  # Not enough frames left for a full chunk

        chunk = inputs[:, start_idx:end_idx, :, :]
        # Resample chunk to have exactly chunk_size frames
        chunk = resample_fixed(chunk, chunk_size=chunk_size)
        chunked_inputs.append(chunk)

    if len(chunked_inputs) == 0:
        raise ValueError("No chunks created from input frames")

    return chunked_inputs


def extract_features_for_video(video_path,
                               preprocess,  chunk_size, stack_size, step_size,
                               model, device, batch_size):
    """Extract features from a video file.

    Args:
        video_path: str, path to video file
        preprocess: preprocessing function
        chunk_size: int, number of frames per chunk after resampling
        stack_size: int, number of frames to stack before resampling
        step_size: int, step size for moving the window
        model: feature extraction model
        device: torch device
        batch_size: int, batch size for processing chunks

    Returns:
        np.ndarray of shape (N_chunks, feature_dim)
    """

    # --- load video ---
    video_data, _, info = io.read_video(
        filename=video_path,
        pts_unit='sec',
        output_format="TCHW"
    )
    assert video_data.ndim == 4  # T, C, H, W
    assert video_data.shape[1] == 3  # C=3

    # --- preprocess ---
    inputs = preprocess(video_data)  # C, T, H, W
    assert inputs.ndim == 4
    assert inputs.shape[0] == 3  # C=3

    # --- chunk frames ---
    chunked_inputs = chunk_frames(inputs,
                                  chunk_size=chunk_size,
                                  stack_size=stack_size,
                                  step_size=step_size)
    assert len(chunked_inputs) > 0

    # --- extract features in batches ---
    all_feats = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(0, len(chunked_inputs), batch_size):
            batch_chunks = chunked_inputs[i:i+batch_size]
            # (B, C, chunk_size, H, W)
            batch_tensor = torch.stack(batch_chunks).to(device)

            # inference
            outputs = model(batch_tensor)  # (B, feature_dim)
            outputs = outputs.cpu().numpy().astype(np.float32)

            all_feats.append(outputs)

    if len(all_feats) == 0:
        raise ValueError("No features extracted from video")

    feats = np.vstack(all_feats)  # shape (N_chunks, feature_dim)
    return feats


# --- load pretrained model + transforms ---
print(f">> Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = MViT_V2_S_Weights.DEFAULT
preprocess = weights.transforms()
model = mvit_v2_s(weights=weights).to(device)
# remove head + norm to get feature vectors
model.head = torch.nn.Identity()
model.norm = torch.nn.Identity()
model.eval()


# --- get all video paths ---
video_paths = sorted(Path("./raw_video/").glob("*.avi"))
assert len(video_paths) == 1970


# --- open HDF5 and process videos ---
feature_file_path = "./MSVD_MViTv2.hdf5"
chunk_size = 16
stack_size = 40
step_size = 32
print(f">> Extracting feature...")
with h5py.File(feature_file_path, "w") as hf:
    for vp in tqdm(video_paths, desc="Videos"):
        # extract features
        feats = extract_features_for_video(
            video_path=str(vp),
            preprocess=preprocess,
            chunk_size=chunk_size,
            stack_size=stack_size,
            step_size=step_size,
            model=model,
            device=device,
            batch_size=8
        )  # (N_chunks, feature_dim)

        # save to HDF5
        hf.create_dataset(vp.stem, data=feats)
print("Done.")

# --- verification ---
print(f">> Verifying saved HDF5 file...")
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
print("Done!")
