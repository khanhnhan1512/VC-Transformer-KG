# coding=utf-8
from __future__ import print_function, division

import os
import re
import random
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Tuple
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import T5TokenizerFast, CLIPImageProcessor

from loader.keyframe import extract_gop_data


VIDEO_EXTENSIONS = ['.avi', '.mp4', '.mkv', '.webm']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def normalize_frames(frames: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 -> (B, T, 3, H, W) float, normalized with CLIP mean/std."""
    pixel_values = frames.float().div_(255.0)
    pixel_values = pixel_values.permute(0, 1, 4, 2, 3).contiguous()
    mean = mean.view(1, 1, 3, 1, 1)
    std = std.view(1, 1, 3, 1, 1)
    return (pixel_values - mean) / std


class E2EDataset(Dataset):
    """Video-caption pairs; mỗi video = chuỗi GOP (I-frame + MV map + timestamp)."""

    def __init__(self, C, phase, caption_fpath):
        self.C = C
        self.phase = phase
        self.caption_fpath = caption_fpath

        # vid -> (frames (T,size,size,3) uint8, motion_maps (T,2,g,g) float32,
        #         timestamps (T,) float32, num_real)
        self.video_frames: Dict[str, Tuple] = {}
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
            f"[E2EDataset] Video file for vid '{vid}' not found in {folder} "
            f"(tried extensions: {VIDEO_EXTENSIONS})")

    def _cache_fpath(self) -> str:
        cache_dpath = getattr(self.C.loader, 'frame_cache_dpath', "")
        if not cache_dpath:
            return ""
        threshold = self.C.loader.keyframe_threshold
        grid = self.C.transformer.motion_grid_size
        fname = f"{self.C.corpus}_{self.phase}_gop{threshold}_g{grid}.npz"
        return os.path.join(cache_dpath, fname)

    def load_video_frames(self):
        """Extract GOP data once per video (for vids appearing in captions)."""
        threshold = self.C.loader.keyframe_threshold
        grid_size = self.C.transformer.motion_grid_size

        cache_fpath = self._cache_fpath()
        if cache_fpath and os.path.exists(cache_fpath):
            print(f"[E2EDataset] Loading cached GOP data from {cache_fpath}")
            cached = np.load(cache_fpath)
            vids = cached['vids']
            for i, vid in enumerate(vids):
                self.video_frames[str(vid)] = (
                    cached['frames'][i], cached['motion_maps'][i],
                    cached['timestamps'][i], int(cached['num_reals'][i]))
            return

        vids = list(self.captions.keys())
        for vid in tqdm(vids, desc=f'Extract_gop_data ({self.phase})'):
            video_fpath = self._find_video_fpath(vid)
            frames, motion_maps, timestamps, num_real = extract_gop_data(
                video_fpath, threshold=threshold, grid_size=grid_size)
            self.video_frames[vid] = (frames, motion_maps, timestamps, num_real)

        if cache_fpath:
            os.makedirs(os.path.dirname(cache_fpath), exist_ok=True)
            np.savez_compressed(
                cache_fpath,
                vids=np.array(vids),
                frames=np.stack([self.video_frames[v][0] for v in vids], axis=0),
                motion_maps=np.stack([self.video_frames[v][1] for v in vids], axis=0),
                timestamps=np.stack([self.video_frames[v][2] for v in vids], axis=0),
                num_reals=np.array([self.video_frames[v][3] for v in vids]),
            )
            print(f"[E2EDataset] Saved GOP cache to {cache_fpath}")

    def build_video_caption_pairs(self):
        self.load_captions()
        self.load_video_frames()

        for vid in self.video_frames.keys():
            frames, motion_maps, timestamps, num_real = self.video_frames[vid]
            for caption in self.captions[vid]:
                self.data.append((vid, frames, motion_maps, timestamps, num_real, caption))

    def __getitem__(self, idx):
        vid, frames, motion_maps, timestamps, num_real, caption = self.data[idx]
        return vid, frames, motion_maps, timestamps, num_real, caption


class E2ECorpus:

    def __init__(self, C, dataset_cls=E2EDataset):
        self.C = C
        self.tokenizer = T5TokenizerFast.from_pretrained(C.transformer.t5_model_name)
        self.E2EDataset = dataset_cls

        image_processor = CLIPImageProcessor.from_pretrained(C.transformer.clip_model_name)
        self.image_mean = torch.tensor(image_processor.image_mean)
        self.image_std = torch.tensor(image_processor.image_std)

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
        dataset = self.E2EDataset(
            self.C,
            phase,
            caption_fpath,
        )
        return dataset

    def frame_collate_fn(self, batch):
        vids, frames, motion_maps, timestamps, num_reals, captions = zip(*batch)

        frames = torch.from_numpy(np.stack(frames, axis=0))  # (B, T, H, W, 3) uint8
        pixel_values = normalize_frames(frames, self.image_mean, self.image_std)

        motion_maps = torch.from_numpy(np.stack(motion_maps, axis=0)).float()  # (B, T, 2, g, g)
        timestamps = torch.from_numpy(np.stack(timestamps, axis=0)).float()    # (B, T)

        # Attention mask over GOP positions: 1 = real GOP, 0 = padding.
        # Built from num_real, NOT inferred from pixels (a black padded frame
        # still yields a non-zero embedding after CLIP).
        T = pixel_values.size(1)
        num_reals = torch.tensor(num_reals, dtype=torch.long)
        frame_mask = (torch.arange(T).unsqueeze(0) < num_reals.unsqueeze(1)).long()

        tokenized = self.tokenizer(
            list(captions),
            max_length=self.C.transformer.max_caption_tokens,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )

        return vids, (pixel_values, motion_maps, timestamps, frame_mask), \
            tokenized.input_ids, tokenized.attention_mask, list(captions)

    def build_data_loader(self, dataset):
        g = torch.Generator()
        data_loader = DataLoader(
            dataset,
            batch_size=self.C.batch_size,
            shuffle=False,
            sampler=RandomSampler(dataset, replacement=False),
            num_workers=self.C.loader.num_workers,
            collate_fn=self.frame_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return data_loader
