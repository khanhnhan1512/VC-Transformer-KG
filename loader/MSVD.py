# coding=utf-8
import pandas as pd
from loader.data_loader_fusion import CustomDataset, Corpus


class MSVDDataset(CustomDataset):
    """ MSVD Dataset """

    def load_captions(self):
        df = pd.read_csv(self.caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[['VideoID', 'Start', 'End', 'Description']]
        df = df[pd.notnull(df['Description'])]

        for video_id, start, end, caption in df.values:
            vid = "{}_{}_{}".format(video_id, start, end)
            self.captions[vid].append(self.preprocess_caption(caption))


class MSVD(Corpus):
    """ MSVD Corpus """

    def __init__(self, C):
        super(MSVD, self).__init__(C, MSVDDataset)
