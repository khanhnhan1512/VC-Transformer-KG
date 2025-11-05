from __future__ import print_function, division

from collections import defaultdict

from tqdm import tqdm
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms

from loader.transform import UniformSample, RandomSample, ToTensor, TrimExceptAscii, Lowercase, \
    RemovePunctuation, SplitWithWhiteSpace, Truncate, PadFirst, PadLast, PadToLength, \
    ToIndex

import random


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

        self.image_video_feats = defaultdict(lambda: [])
        self.motion_video_feats = defaultdict(lambda: [])
        self.object_video_feats = defaultdict(lambda: [])
        self.rel_feats = defaultdict(lambda: [])

        # captions: {vid, caption}
        self.r2l_captions = defaultdict(lambda: [])
        self.l2r_captions = defaultdict(lambda: [])
        self.data = []

        self.build_video_caption_pairs()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid, image_video_feats, motion_video_feats, object_video_feats, rel_feats, r2l_caption, l2r_caption = \
            self.data[idx]
        if self.transform_frame:
            image_video_feats = [self.transform_frame(feat) for feat in image_video_feats]
            motion_video_feats = [self.transform_frame(feat) for feat in motion_video_feats]
            object_video_feats = [self.transform_frame(feat) for feat in object_video_feats]
            rel_feats = [self.transform_frame(feat) for feat in rel_feats]
        if self.transform_caption:
            r2l_caption = self.transform_caption(r2l_caption)
            l2r_caption = self.transform_caption(l2r_caption)

        return vid, image_video_feats, motion_video_feats, object_video_feats, rel_feats, r2l_caption, l2r_caption

    # It's too complex so write another function
    def load_object_feats(self, frames, fin_o, fin_b, vid):
        raise NotImplementedError("[CustomDataset.load_object_feats] You should implement this function.")
        """
        # vid = 'video122'
        # assert vid == vid1, "video id of OFeat and BFeat is not align"
        feats_b = fin_b[vid][()]
        feats_o = fin_o[vid][()]
        # to = torch.Tensor(feats_o).cuda()
        # tb = torch.Tensor(feats_b).cuda()
        # feats = torch.cat((tb, to), 1).cpu().numpy()
        # feats = np.hstack((feats_b, feats_o))
        feats = np.concatenate((feats_b, feats_o), axis=1)
        num_paddings = frames - len(feats)
        if feats.size == 0:
            # now just object feat may appear the feature is empty
            feats = np.zeros((frames, 1028))
        else:
            feats = feats.tolist() + [np.zeros_like(feats[0])
                                      for _ in range(num_paddings)]
        feats = np.asarray(feats)
        sampled_idxs = np.linspace(
            0, len(feats) - 1, frames, dtype=int)  # return evenly sapced number within the specified
        feats = feats[sampled_idxs]
        assert len(feats) == frames
        return feats
        """;

    def load_four_video_feats(self):
        models = self.C.feat.model.split('+')
        print('Enter the load4 method.')
        for i in range(len(models)):
            # print('Begin to start load %d feats, total are %d' % (i + 1, len(models)))
            frames = self.C.loader.frame_sample_len
            # i = 2
            # if i == 2:
            #     frames = self.C.feat.num_boxes
            # if i == 3:
            #     frames = self.C.feat.three_turple
            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(
                "MSVD", "MSVD" + '_' + models[i], self.phase)
            fpath_b = self.C.loader.phase_video_feat_fpath_tpl.format(
                "MSVD", "MSVD" + '_' + 'BFeat', self.phase)  # load two feats at the sames
            # time, there are some problems in efficiency
            fin = h5py.File(fpath, 'r')
            fin_b = h5py.File(fpath_b, 'r')
            tqdm(fin.keys()).set_description('Load_four_feature_feats:')
            for vid in tqdm(fin.keys()):
                # vid = 'video122'
                feats = fin[vid][()]
                if len(feats) < frames:
                    """
                    if i == 2:
                        # fin_o = h5py.File(fpath, 'r')

                        feats = self.load_object_feats(
                            frames=frames, fin_o=fin, fin_b=fin_b, vid=vid)
                        self.object_video_feats[vid].append(feats)
                        continue
                    """;
                    num_paddings = frames - len(feats)
                    if feats.size == 0:
                        raise ValueError("[CustomDataset.load_four_video_feats] Feature size is zero!")
                        """
                        # for _ in range(num_paddings):
                        # now just object feat may appear the feature is empty
                        feats = np.zeros((frames, 1024))
                        """;
                    else:
                        feats = feats.tolist() + [np.zeros_like(feats[0])
                                                  for _ in range(num_paddings)]
                    # feats = feats.tolist() + [np.zeros_like(feats[0])
                    #                           for _ in range(num_paddings)]
                    feats = np.asarray(feats)
                    sampled_idxs = np.linspace(
                        0, len(feats) - 1, frames, dtype=int)  # return evenly sapced number within the specified
                    feats = feats[sampled_idxs]
                    assert len(feats) == frames
                    if i == 0:
                        self.image_video_feats[vid].append(feats)
                    elif i == 1:
                        self.motion_video_feats[vid].append(feats)
                    elif i == 2:
                        self.object_video_feats[vid].append(feats)
                    elif i == 3:
                        self.rel_feats[vid].append(feats)
                else:
                    if i == 0:
                        self.image_video_feats[vid].append(feats)
                    elif i == 1:
                        self.motion_video_feats[vid].append(feats)
                    elif i == 2:
                        """
                        feats = self.load_object_feats(
                            frames=frames, fin_o=fin, fin_b=fin_b, vid=vid)
                        """;
                        self.object_video_feats[vid].append(feats)
                    elif i == 3:
                        self.rel_feats[vid].append(feats)
            fin.close()
            fin_b.close()

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def build_video_caption_pairs(self):
        self.load_captions()
        self.load_four_video_feats()
        assert self.image_video_feats.keys() == self.motion_video_feats.keys(), "Image feats is not match with motion feats"
        for vid in self.image_video_feats.keys():
            image_video_feats = self.image_video_feats[vid]
            motion_video_feats = self.motion_video_feats[vid]
            if self.object_video_feats[vid]:
                object_video_feats = self.object_video_feats[vid]
            else:
                raise NotImplementedError("Object features are missing!")
                # object_video_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.msrvtt_dim)))
            if self.rel_feats[vid]:
                rel_feats = self.rel_feats[vid]
            else:
                raise NotImplementedError("Relation features are missing!")
                # rel_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.rel_dim)))
                # self.C.FeatureConfig.size[-1]
            for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                self.data.append((vid, image_video_feats, motion_video_feats, object_video_feats, rel_feats,
                                  r2l_caption, l2r_caption))


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
            Sample(self.C.loader.frame_sample_len),
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

        self.train_data_loader = self.build_data_loader(self.train_dataset, phase='train')
        self.val_data_loader = self.build_data_loader(self.val_dataset, phase='val')
        self.test_data_loader = self.build_data_loader(self.test_dataset, phase='test')

    def build_dataset(self, phase, caption_fpath):
        dataset = self.CustomDataset(
            self.C,
            phase,
            caption_fpath,
            transform_frame=self.transform_frame,
            transform_caption=self.transform_caption,
        )
        return dataset

    def four_feature_collate_fn(self, batch):
        vids, image_video_feats, motion_video_feats, object_video_feats, rel_feats, r2l_captions, l2r_captions = zip(
            *batch)
        image_video_feats_list = zip(*image_video_feats)
        motion_video_feats_list = zip(*motion_video_feats)
        object_video_feats_list = zip(*object_video_feats)
        rel_feats_list = zip(*rel_feats)

        image_video_feats_list = [torch.stack(video_feats).float() for video_feats in image_video_feats_list]
        motion_video_feats_list = [torch.stack(video_feats).float() for video_feats in motion_video_feats_list]
        object_video_feats_list = [torch.stack(video_feats).float() for video_feats in object_video_feats_list]
        rel_feats_list = [torch.stack(video_feats).float() for video_feats in rel_feats_list]

        r2l_captions = torch.stack(r2l_captions)
        l2r_captions = torch.stack(l2r_captions)

        r2l_captions = r2l_captions.float()
        l2r_captions = l2r_captions.float()
        return vids, image_video_feats_list, motion_video_feats_list, object_video_feats_list, rel_feats_list, r2l_captions, l2r_captions

    def build_data_loader(self, dataset, phase):
        collate_fn = self.four_feature_collate_fn

        if phase == 'test':
            batch_size = 1
        else:
            batch_size = self.C.batch_size

        g = torch.Generator()
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # If sampler is specified, shuffle must be False.
            sampler=RandomSampler(dataset, replacement=False),
            num_workers=self.C.loader.num_workers,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return data_loader
