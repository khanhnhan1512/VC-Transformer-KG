from __future__ import print_function, division


import random
import h5py
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler

from loader.transform import UniformSample, RandomSample, ToTensor, TrimExceptAscii, Lowercase, \
    RemovePunctuation, SplitWithWhiteSpace, Truncate, PadFirst, PadLast, PadToLength, \
    ToIndex


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomVocab:
    def __init__(self, caption_fpath, init_word2idx, min_count, transform):
        self.caption_fpath = caption_fpath
        self.min_count = min_count
        self.transform = transform

        self.word2idx = init_word2idx
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq_dict = defaultdict(lambda: 0)
        self.n_vocabs = len(self.word2idx)
        self.n_words = self.n_vocabs
        self.max_sentence_len = -1

        self.build()

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def build(self):
        raise NotImplementedError("You should implement this function.")


class CustomDataset(Dataset):
    """ Dataset """

    def __init__(self, C, phase, caption_fpath, transform_caption, transform_frame):
        self.C = C
        self.phase = phase
        self.caption_fpath = caption_fpath
        self.transform_frame = transform_frame
        self.transform_caption = transform_caption

        # features: {vid, [feature1, feature2, ...]}
        self.video_features: Dict[str, List[np.ndarray]] = defaultdict(lambda: [])
        self.num_features: int = -1 # Updated when loading features

        # captions: {vid, caption}
        self.r2l_captions = defaultdict(lambda: [])
        self.l2r_captions = defaultdict(lambda: [])
        self.data = []

        self.build_video_caption_pairs()

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def __len__(self):
        return len(self.data)

    def load_video_feats(self):
        print('Enter the load_video_feats method.')
        models = self.C.feat.model.split('+')
        self.num_features = len(models)
        
        for i in range(self.num_features):
            num_tokens: int = self.C.loader.frame_sample_len
            
            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(
                self.C.corpus, self.C.corpus + '_' + models[i], self.phase)
            
            # time, there are some problems in efficiency
            fin = h5py.File(fpath, 'r')
            
            tqdm(fin.keys()).set_description('Load_feature_feats:')
            for vid in tqdm(fin.keys()):
                feature: np.ndarray = fin[vid][()]
                if feature.size == 0: raise ValueError("[CustomDataset.load_video_feats] Feature size is zero!")
                
                if len(feature) < num_tokens: # zero padding
                    num_padding = num_tokens - feature.shape[0]
                    pad_tokens = np.zeros((num_padding, feature.shape[1]), dtype=feature.dtype)
                    feature = np.concatenate((feature, pad_tokens), axis=0)
                elif len(feature) > num_tokens:
                    sampled_idxs = np.linspace(0, len(feature) - 1, num_tokens, dtype=int)  # return evenly sapced number within the specified
                    feature = feature[sampled_idxs]
                assert len(feature) == num_tokens
                
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
            
            for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                self.data.append((vid, feature_list, r2l_caption, l2r_caption))

    def __getitem__(self, idx):
        vid, feature_list, r2l_caption, l2r_caption = self.data[idx]
        
        if self.transform_frame: # transform video features
            feature_list = [self.transform_frame(feat) for feat in feature_list]

        if self.transform_caption: # transform captions
            r2l_caption = self.transform_caption(r2l_caption)
            l2r_caption = self.transform_caption(l2r_caption)

        return vid, *feature_list, r2l_caption, l2r_caption


class Corpus:
    """ Data Loader """

    def __init__(self, C, vocab_cls=CustomVocab, dataset_cls=CustomDataset):
        self.C = C
        self.vocab = None
        self.train_dataset = None
        self.train_data_loader = None
        self.val_dataset = None
        self.val_data_loader = None
        self.test_dataset = None
        self.test_data_loader = None

        self.CustomVocab = vocab_cls
        self.CustomDataset = dataset_cls

        self.transform_sentence = transforms.Compose([
            TrimExceptAscii(),
            Lowercase(),
            RemovePunctuation(),
            SplitWithWhiteSpace(),
            Truncate(self.C.loader.max_caption_len),
        ])

        self.build()

    def build(self):
        self.build_vocab()
        self.build_data_loaders()

    def build_vocab(self):
        self.vocab = self.CustomVocab(
            self.C.loader.train_caption_fpath,
            self.C.vocab.init_word2idx,
            self.C.loader.min_count,
            transform=self.transform_sentence)

    def build_data_loaders(self):
        """ Transformation """
        if self.C.loader.frame_sampling_method == "uniform":
            Sample = UniformSample
        elif self.C.loader.frame_sampling_method == "random":
            Sample = RandomSample
        else:
            raise NotImplementedError("Unknown frame sampling method: {}".format(self.C.loader.frame_sampling_method))

        self.transform_frame = transforms.Compose([
            #Sample(self.C.loader.frame_sample_len),
            ToTensor(torch.float),
        ])
        self.transform_caption = transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.word2idx),
            PadFirst(self.vocab.word2idx['<S>']),
            PadLast(self.vocab.word2idx['<S>']),
            # +2 for <SOS> and <EOS>
            PadToLength(self.vocab.word2idx['<PAD>'],
                        self.vocab.max_sentence_len + 2),
            ToTensor(torch.long),
        ])

        self.train_dataset = self.build_dataset("train", self.C.loader.train_caption_fpath)
        self.val_dataset = self.build_dataset("val", self.C.loader.val_caption_fpath)
        self.test_dataset = self.build_dataset("test", self.C.loader.test_caption_fpath)

        self.train_data_loader = self.build_data_loader(self.train_dataset)
        self.val_data_loader = self.build_data_loader(self.val_dataset)
        self.test_data_loader = self.build_data_loader(self.test_dataset)

    def build_dataset(self, phase, caption_fpath):
        dataset = self.CustomDataset(
            self.C,
            phase,
            caption_fpath,
            transform_frame=self.transform_frame,
            transform_caption=self.transform_caption,
        )
        return dataset

    def feature_collate_fn(self, batch):
        vids, *features, r2l_captions, l2r_captions = zip(*batch)

        features_list = [torch.stack(feats_tuple) for feats_tuple in features]
        r2l_captions = torch.stack(r2l_captions)
        l2r_captions = torch.stack(l2r_captions)
        
        return vids, features_list, r2l_captions, l2r_captions

    def build_data_loader(self, dataset):
        g = torch.Generator()
        data_loader = DataLoader(
            dataset,
            batch_size=self.C.batch_size,
            shuffle=False,  # If sampler is specified, shuffle must be False.
            sampler=RandomSampler(dataset, replacement=False),
            num_workers=self.C.loader.num_workers,
            collate_fn=self.feature_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return data_loader
