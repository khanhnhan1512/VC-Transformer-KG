# coding=utf-8
import json
import random
from loader.data_loader_fusion import CustomVocab, CustomDataset, Corpus
from typing import List


class VATEXVocab(CustomVocab):
    """ VATEX Vocaburary """

    def load_captions(self):
        with open(self.caption_fpath, 'r') as fin:
            data = json.load(fin)

        captions: List[str] = []
        for item in data:
            assert type(item['enCap']) == list, f"[VATEXVocab.load_captions] {item['videoID']} has {type(item['enCap'])} captions, expected list."
            assert len(item['enCap']) == 10, f"[VATEXVocab.load_captions] {item['videoID']} has {len(item['enCap'])} captions, expected 10."
            captions += item['enCap']
        return captions

    def build(self):
        captions = self.load_captions()
        for caption in captions:
            words = self.transform(caption)
            self.max_sentence_len = max(self.max_sentence_len, len(words))
            for word in words:
                self.word_freq_dict[word] += 1
        self.n_vocabs_untrimmed = len(self.word_freq_dict)
        self.n_words_untrimmed = sum(list(self.word_freq_dict.values()))

        keep_words = [word for word, freq in self.word_freq_dict.items()
                      if freq >= self.min_count]

        for idx, word in enumerate(keep_words, len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.n_vocabs = len(self.word2idx)
        self.n_words = sum([self.word_freq_dict[word] for word in keep_words])


class VATEXDataset(CustomDataset):
    """ VATEX Dataset """

    def load_captions(self):
        with open(self.caption_fpath, 'r') as fin:
            data = json.load(fin)

        for item in data:
            vid = item['videoID']
            for caption in item['enCap']:
                r2l_caption = " ".join(caption.strip('.').split()[::-1])
                self.r2l_captions[vid].append(r2l_caption)
                self.l2r_captions[vid].append(caption)
        for vid, caption in self.r2l_captions.items():
            # self.r2l_captions[vid] = caption[::-1]
            random.shuffle(caption)


class VATEX(Corpus):
    """ VATEX Corpus """

    def __init__(self, C):
        super(VATEX, self).__init__(C, VATEXVocab, VATEXDataset)
