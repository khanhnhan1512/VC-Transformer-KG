from __future__ import print_function, division

import re
import random
import h5py
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5TokenizerFast

from loader.transform import ToTensor


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomDataset(Dataset):

    def __init__(self, C, phase, caption_fpath, transform_frame):
        self.C = C
        self.phase = phase
        self.caption_fpath = caption_fpath
        self.transform_frame = transform_frame

        self.video_features: Dict[str, List[np.ndarray]] = defaultdict(lambda: [])
        self.num_features: int = -1

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

    def load_video_feats(self):
        print('Enter the load_video_feats method.')
        feature_names = self.C.feat.feature_names
        self.num_features = len(feature_names)

        # Số GOP giữ lại cho mỗi video (pad lên / sample xuống về con số này)
        num_gop: int = self.C.loader.num_gop

        for i in range(self.num_features):
            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(
                self.C.corpus, self.C.corpus + '_' + feature_names[i], self.phase)

            fin = h5py.File(fpath, 'r')

            for vid in tqdm(fin.keys(), desc=f'Load feats [{feature_names[i]}]'):
                feature: np.ndarray = fin[vid][()]
                if feature.size == 0: raise ValueError("[CustomDataset.load_video_feats] Feature size is zero!")

                # Mỗi modality được chuẩn hóa ĐỘC LẬP về đúng num_gop token.
                # KHÔNG còn ràng buộc các modality phải cùng NUM_GOP thô cho cùng
                # 1 video: các feature ở đây có thể đến từ những pipeline trích
                # xuất KHÁC nhau (SigLIP2 theo GOP H.264 vs. feature cũ của BiDecT
                # theo keyframe scheme riêng), nên số frame/GOP thô lệch nhau là
                # bình thường. Hệ quả: token GOP-t của hai modality KHÔNG còn nhất
                # thiết trùng khoảnh khắc thời gian, và mask phải dựng ĐỘC LẬP cho
                # từng modality từ zero-pad của chính nó (xem
                # T5Captioner._build_encoder_attention_mask).
                if len(feature) < num_gop:
                    num_padding = num_gop - feature.shape[0]
                    # shape[1:] để pad đúng với MỌI ndim:
                    #   (N, D)          appearance pre-extracted
                    #   (N, 32, D)      Q-Former tokens
                    #   (N, K, C, G, G) motion vector grid (feature THÔ)
                    pad_shape = (num_padding,) + feature.shape[1:]
                    pad_gops = np.zeros(pad_shape, dtype=feature.dtype)
                    feature = np.concatenate((feature, pad_gops), axis=0)
                elif len(feature) > num_gop:
                    sampled_idxs = np.linspace(0, len(feature) - 1, num_gop, dtype=int)
                    feature = feature[sampled_idxs]
                assert len(feature) == num_gop

                self.video_features[vid].append(feature)

            fin.close()

    def build_video_caption_pairs(self):
        self.load_captions()
        self.load_video_feats()

        for vid in self.video_features.keys():
            feature_list: List[np.ndarray] = self.video_features[vid]
            assert len(feature_list) == self.num_features, \
                f"[CustomDataset.build_video_caption_pairs] Number of features mismatch: " \
                f"expected {self.num_features}, got {len(feature_list)}"

            for caption in self.captions[vid]:
                self.data.append((vid, feature_list, caption))

    def __getitem__(self, idx):
        vid, feature_list, caption = self.data[idx]

        if self.transform_frame:
            feature_list = [self.transform_frame(feat) for feat in feature_list]

        return vid, *feature_list, caption


class Corpus:

    def __init__(self, C, dataset_cls=CustomDataset):
        self.C = C
        self.tokenizer = T5TokenizerFast.from_pretrained(C.transformer.t5_model_name)
        self.CustomDataset = dataset_cls

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
        self.transform_frame = transforms.Compose([
            ToTensor(torch.float),
        ])

        self.train_dataset = self.build_dataset("train", self.C.loader.train_caption_fpath)
        self.val_dataset = self.build_dataset("val", self.C.loader.val_caption_fpath)
        self.test_dataset = self.build_dataset("test", self.C.loader.test_caption_fpath)

        self.train_data_loader = self.build_data_loader(self.train_dataset, shuffle=True)
        self.val_data_loader = self.build_data_loader(self.val_dataset, shuffle=False)
        self.test_data_loader = self.build_data_loader(self.test_dataset, shuffle=False)

    def build_dataset(self, phase, caption_fpath):
        dataset = self.CustomDataset(
            self.C,
            phase,
            caption_fpath,
            transform_frame=self.transform_frame,
        )
        return dataset

    def feature_collate_fn(self, batch):
        vids, *features, captions = zip(*batch)

        features_list = [torch.stack(feats_tuple) for feats_tuple in features]

        tokenized = self.tokenizer(
            list(captions),
            max_length=self.C.transformer.max_caption_tokens,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )

        return vids, features_list, tokenized.input_ids, tokenized.attention_mask, list(captions)

    def build_data_loader(self, dataset, shuffle):
        g = torch.Generator()
        # Chỉ train mới cần xáo trộn. Với val/test, thứ tự KHÔNG ảnh hưởng loss
        # lẫn CIDEr — nhưng RandomSampler (không truyền generator) rút RNG TOÀN
        # CỤC mỗi lần iter, nên số lượt quét loader phía eval sẽ làm dịch luồng
        # RNG và đổi quỹ đạo train ở các epoch sau. Dùng SequentialSampler để
        # eval tất định và tách rời hoàn toàn khỏi RNG của train.
        sampler = (RandomSampler(dataset, replacement=False) if shuffle
                   else SequentialSampler(dataset))
        data_loader = DataLoader(
            dataset,
            batch_size=self.C.batch_size,
            shuffle=False,  # thứ tự do `sampler` quyết định
            sampler=sampler,
            num_workers=self.C.loader.num_workers,
            collate_fn=self.feature_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            # pin_memory: batch nằm ở vùng nhớ ghim -> copy H2D chạy ASYNC
            # (cùng với non_blocking=True trong parse_batch), chồng lấn với
            # compute thay vì chặn luồng chính. KHÔNG đổi giá trị, chỉ đổi lịch.
            pin_memory=True,
            # Giữ worker sống qua các epoch thay vì spawn lại 4 worker mỗi lần
            # iter (train + val, 20 epoch -> hàng chục lần spawn vô ích).
            persistent_workers=self.C.loader.num_workers > 0,
        )
        return data_loader
