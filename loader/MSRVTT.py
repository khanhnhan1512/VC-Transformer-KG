# coding=utf-8
import json
import random
from loader.data_loader_fusion import CustomVocab, CustomDataset, Corpus


class MSRVTTVocab(CustomVocab):
    """ MSR-VTT Vocaburary """

    def load_captions(self):
        with open(self.caption_fpath, 'r') as fin:
            data = json.load(fin)

        captions = []
        for vid, depth1 in data.items():
            for sid, caption in depth1.items():
                captions.append(caption)
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


class MSRVTTDataset(CustomDataset):
    """ MSR-VTT Dataset """

    def load_captions(self):
        with open(self.caption_fpath, 'r') as fin:
            data = json.load(fin)

        for vid, depth1 in data.items():
            for caption in depth1.values():
                r2l_caption = " ".join(caption.strip('.').split()[::-1])
                self.r2l_captions[vid].append(r2l_caption)
                self.l2r_captions[vid].append(caption)
        for vid, caption in self.r2l_captions.items():
            # self.r2l_captions[vid] = caption[::-1]
            random.shuffle(caption)


class MSRVTT(Corpus):
    """ MSR-VTT Corpus """

    def __init__(self, C):
        super(MSRVTT, self).__init__(C, MSRVTTVocab, MSRVTTDataset)
