# coding=utf-8
from __future__ import print_function, division

import os
import re
import random
import h5py
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import T5TokenizerFast

from loader.keyframe import extract_gop_motion


VIDEO_EXTENSIONS = ['.avi', '.mp4', '.mkv', '.webm']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class GOPDataset(Dataset):
    """Video = chuỗi GOP với 2 nguồn dữ liệu thẳng hàng 1-1:

    1. Pre-extracted features từ HDF5 (1 vector / GOP / feature, ví dụ
       Blip2QFormerMeanKF) — trích bởi blip2_extractor.py, đủ độ dài.
    2. MV map trích từ chính file video (1 map / GOP) cho motion encoder
       built-in của model.

    Alignment được ASSERT khi dựng dataset (số GOP của HDF5 == số GOP PyAV
    đọc từ video). Chuẩn hóa độ dài về num_gop được làm CHUNG một chỗ
    (cùng bộ chỉ số linspace / cùng lượng pad) nên vị trí padding của mọi
    modality trùng nhau — mask chỉ cần suy từ feature anchor (BLIP-2).
    """

    def __init__(self, C, phase, caption_fpath):
        self.C = C
        self.phase = phase
        self.caption_fpath = caption_fpath

        # vid -> (feats_list [(T, D_m)], motion (T, 2, g, g), timestamps (T,), num_real)
        self.video_data: Dict[str, Tuple] = {}
        self.captions = defaultdict(lambda: [])
        self.data = []

        self.build_video_caption_pairs()

    @staticmethod
    def preprocess_caption(caption):
        caption = caption.encode('ascii', 'ignore').decode('ascii')
        caption = caption.lower()
        caption = re.sub(r'[^\w\s]', '', caption)
        return ' '.join(caption.split())

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def __len__(self):
        return len(self.data)

    def _find_video_fpath(self, vid: str) -> str:
        folder = self.C.loader.VIDEO_FOLDER_PATH
        for ext in VIDEO_EXTENSIONS:
            fpath = os.path.join(folder, vid + ext)
            if os.path.exists(fpath):
                return fpath
        raise FileNotFoundError(
            f"[GOPDataset] Video file for vid '{vid}' not found in {folder} "
            f"(tried extensions: {VIDEO_EXTENSIONS})")

    def _mv_cache_fpath(self) -> str:
        cache_dpath = getattr(self.C.loader, 'mv_cache_dpath', "")
        if not cache_dpath:
            return ""
        grid = self.C.transformer.motion_grid_size
        fname = f"{self.C.corpus}_{self.phase}_mvraw_g{grid}.npy"
        return os.path.join(cache_dpath, fname)

    def _load_motion_raw(self, vids: List[str]) -> Dict[str, Tuple]:
        """vid -> (motion_maps (L,2,g,g), timestamps (L,)) — đủ độ dài, chưa normalize."""
        grid = self.C.transformer.motion_grid_size

        cache_fpath = self._mv_cache_fpath()
        if cache_fpath and os.path.exists(cache_fpath):
            print(f"[GOPDataset] Loading cached MV maps from {cache_fpath}")
            return np.load(cache_fpath, allow_pickle=True).item()

        motion_raw = {}
        for vid in tqdm(vids, desc=f'Extract_gop_motion ({self.phase})'):
            video_fpath = self._find_video_fpath(vid)
            motion_maps, timestamps, _ = extract_gop_motion(video_fpath, grid_size=grid)
            motion_raw[vid] = (motion_maps, timestamps)

        if cache_fpath:
            os.makedirs(os.path.dirname(cache_fpath), exist_ok=True)
            np.save(cache_fpath, motion_raw, allow_pickle=True)
            print(f"[GOPDataset] Saved MV cache to {cache_fpath}")
        return motion_raw

    @staticmethod
    def _normalize_length(arrays: List[np.ndarray], threshold: int):
        """Chuẩn hóa độ dài CHUNG cho mọi modality của 1 video.

        Mọi array cùng độ dài L theo trục 0. L > threshold -> cùng một bộ chỉ
        số linspace; L < threshold -> cùng lượng zero-pad. Nhờ đó vị trí
        pad/mask của mọi modality trùng nhau tuyệt đối.
        """
        L = arrays[0].shape[0]
        if L > threshold:
            idxs = np.linspace(0, L - 1, threshold, dtype=int)
            return [a[idxs] for a in arrays], threshold

        num_real = L
        padded = []
        for a in arrays:
            pad = np.zeros((threshold - L, *a.shape[1:]), dtype=a.dtype)
            padded.append(np.concatenate([a, pad], axis=0))
        return padded, num_real

    def load_video_data(self):
        C = self.C
        threshold = C.loader.num_gop
        grid = C.transformer.motion_grid_size
        use_motion = C.transformer.use_motion_tokens
        models = C.feat.model.split('+')

        h5_files = []
        for m in models:
            fpath = C.loader.phase_video_feat_fpath_tpl.format(
                C.corpus, C.corpus + '_' + m, self.phase)
            h5_files.append(h5py.File(fpath, 'r'))

        vids = list(self.captions.keys())
        motion_raw = self._load_motion_raw(vids) if use_motion else None

        mismatches = []
        for vid in tqdm(vids, desc=f'Load_gop_data ({self.phase})'):
            feats = [hf[vid][()].astype(np.float32) for hf in h5_files]

            L = feats[0].shape[0]
            if any(f.shape[0] != L for f in feats):
                mismatches.append((vid, 'hdf5', [f.shape[0] for f in feats]))
                continue

            if use_motion:
                motion_maps, timestamps = motion_raw[vid]
                # ASSERT alignment: số GOP của HDF5 phải bằng số GOP PyAV đọc từ video
                if motion_maps.shape[0] != L:
                    mismatches.append((vid, 'hdf5-vs-video', (L, motion_maps.shape[0])))
                    continue
            else:
                # Ablation không motion: không cần video folder, dummy zeros
                motion_maps = np.zeros((L, 2, grid, grid), dtype=np.float32)
                timestamps = np.zeros((L,), dtype=np.float32)

            arrays, num_real = self._normalize_length(
                [*feats, motion_maps, timestamps], threshold)
            *feats_n, motion_n, ts_n = arrays
            self.video_data[vid] = (feats_n, motion_n, ts_n, num_real)

        for hf in h5_files:
            hf.close()

        if mismatches:
            detail = "\n".join(f"  {vid} [{kind}]: {info}" for vid, kind, info in mismatches[:20])
            raise ValueError(
                f"[GOPDataset] {len(mismatches)} video lệch alignment giữa các nguồn "
                f"(HDF5 features / video GOP):\n{detail}")

    def build_video_caption_pairs(self):
        self.load_captions()
        self.load_video_data()

        for vid in self.video_data.keys():
            feats, motion_maps, timestamps, num_real = self.video_data[vid]
            for caption in self.captions[vid]:
                self.data.append((vid, feats, motion_maps, timestamps, caption))

    def __getitem__(self, idx):
        return self.data[idx]


class GOPCorpus:

    def __init__(self, C, dataset_cls=GOPDataset):
        self.C = C
        self.tokenizer = T5TokenizerFast.from_pretrained(C.transformer.t5_model_name)
        self.GOPDataset = dataset_cls

        self.train_dataset = None
        self.train_data_loader = None
        self.val_dataset = None
        self.val_data_loader = None
        self.test_dataset = None
        self.test_data_loader = None

        self.build()

    def build(self):
        self.build_data_loaders()

    def build_data_loaders(self):
        self.train_dataset = self.build_dataset("train", self.C.loader.train_caption_fpath)
        self.val_dataset = self.build_dataset("val", self.C.loader.val_caption_fpath)
        self.test_dataset = self.build_dataset("test", self.C.loader.test_caption_fpath)

        self.train_data_loader = self.build_data_loader(self.train_dataset)
        self.val_data_loader = self.build_data_loader(self.val_dataset)
        self.test_data_loader = self.build_data_loader(self.test_dataset)

    def build_dataset(self, phase, caption_fpath):
        return self.GOPDataset(self.C, phase, caption_fpath)

    def gop_collate_fn(self, batch):
        vids, feats, motion_maps, timestamps, captions = zip(*batch)

        # feats: tuple các list [(T, D_m)] -> mỗi modality 1 tensor (B, T, D_m)
        num_modalities = len(feats[0])
        features_list = [
            torch.from_numpy(np.stack([f[m] for f in feats], axis=0))
            for m in range(num_modalities)
        ]
        motion_maps = torch.from_numpy(np.stack(motion_maps, axis=0))  # (B, T, 2, g, g)
        timestamps = torch.from_numpy(np.stack(timestamps, axis=0))    # (B, T)

        tokenized = self.tokenizer(
            list(captions),
            max_length=self.C.transformer.max_caption_tokens,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )

        # Mask không cần truyền: model tự suy từ feature anchor (BLIP-2) —
        # vector thật của QFormer không bao giờ toàn 0, padding thì đúng bằng 0
        return vids, (*features_list, motion_maps, timestamps), \
            tokenized.input_ids, tokenized.attention_mask, list(captions)

    def build_data_loader(self, dataset):
        g = torch.Generator()
        data_loader = DataLoader(
            dataset,
            batch_size=self.C.batch_size,
            shuffle=False,
            sampler=RandomSampler(dataset, replacement=False),
            num_workers=self.C.loader.num_workers,
            collate_fn=self.gop_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return data_loader
