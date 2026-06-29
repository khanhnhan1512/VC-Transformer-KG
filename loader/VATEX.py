# coding=utf-8
import json
from loader.data_loader_fusion import CustomDataset, Corpus
from typing import List


class VATEXDataset(CustomDataset):
    """ VATEX Dataset """

    def load_captions(self):
        with open(self.caption_fpath, 'r') as fin:
            data = json.load(fin)

        for item in data:
            vid = item['videoID']
            for caption in item['enCap']:
                self.captions[vid].append(self.preprocess_caption(caption))


class VATEX(Corpus):
    """ VATEX Corpus """

    def __init__(self, C):
        super(VATEX, self).__init__(C, VATEXDataset)
