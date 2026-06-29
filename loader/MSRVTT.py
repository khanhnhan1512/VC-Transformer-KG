# coding=utf-8
import json
from loader.data_loader_fusion import CustomDataset, Corpus


class MSRVTTDataset(CustomDataset):
    """ MSR-VTT Dataset """

    def load_captions(self):
        with open(self.caption_fpath, 'r') as fin:
            data = json.load(fin)

        for vid, depth1 in data.items():
            for caption in depth1.values():
                self.captions[vid].append(self.preprocess_caption(caption))


class MSRVTT(Corpus):
    """ MSR-VTT Corpus """

    def __init__(self, C):
        super(MSRVTT, self).__init__(C, MSRVTTDataset)
