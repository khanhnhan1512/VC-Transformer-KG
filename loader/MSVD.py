# coding=utf-8
import pandas as pd
from loader.data_loader_fusion import CustomDataset, Corpus
from loader.data_loader_e2e import E2EDataset, E2ECorpus


def _load_msvd_captions(dataset):
    df = pd.read_csv(dataset.caption_fpath)
    df = df[df['Language'] == 'English']
    df = df[['VideoID', 'Start', 'End', 'Description']]
    df = df[pd.notnull(df['Description'])]

    for video_id, start, end, caption in df.values:
        vid = "{}_{}_{}".format(video_id, start, end)
        dataset.captions[vid].append(dataset.preprocess_caption(caption))


class MSVDDataset(CustomDataset):
    """ MSVD Dataset (pre-extracted features pipeline) """

    def load_captions(self):
        _load_msvd_captions(self)


class MSVD(Corpus):
    """ MSVD Corpus (pre-extracted features pipeline) """

    def __init__(self, C):
        super(MSVD, self).__init__(C, MSVDDataset)


class MSVDE2EDataset(E2EDataset):
    """ MSVD Dataset (end-to-end pipeline: keyframes -> CLIP ViT) """

    def load_captions(self):
        _load_msvd_captions(self)


class MSVDE2E(E2ECorpus):
    """ MSVD Corpus (end-to-end pipeline) """

    def __init__(self, C):
        super(MSVDE2E, self).__init__(C, MSVDE2EDataset)
