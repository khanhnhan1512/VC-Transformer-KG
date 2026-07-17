# coding=utf-8
"""Reward CIDEr-D cho SCST, với document frequency tính trên TOÀN BỘ tập train.

Vì sao không dùng thẳng pycocoevalcap.Cider: compute_score() của nó tính df
từ chính batch refs đưa vào — với batch SCST nhỏ, idf sai lệch nặng (và
compute_cider còn ghi đè ref_len = log(batch_size) nên không thể chỉ inject df).
Chuẩn SCST (Rennie 2017; ruotianluo/self-critical.pytorch) là df cố định trên
toàn corpus train. Ở đây vendor lại phần tính CIDEr-D (~80 dòng, bám sát
cider_scorer.py của pycocoevalcap) với 2 khác biệt:
  1. df + log(num_docs) tính MỘT lần từ toàn bộ refs train (document = 1 video).
  2. Vector n-gram của mọi reference được precompute -> mỗi bước SCST chỉ phải
     cook phần hypothesis (nhanh hơn nhiều lần).

Lưu ý nhất quán: evaluate() cuối vẫn dùng pycocoevalcap như cũ — reward này chỉ
dùng làm tín hiệu train, các con số báo cáo không đổi cách đo.
"""
from collections import defaultdict
from typing import Dict, List

import numpy as np
from pycocoevalcap.cider.cider_scorer import precook


class CiderDReward:
    """vid2refs: {video_id: [ref_caption, ...]} của TẬP TRAIN (đã preprocess)."""

    def __init__(self, vid2refs: Dict[str, List[str]], n: int = 4, sigma: float = 6.0):
        self.n = n
        self.sigma = sigma

        # --- Document frequency: document = tập refs của 1 video ---
        self.doc_freq = defaultdict(float)
        cooked_refs = {}  # vid -> list[dict ngram->count]
        for vid, refs in vid2refs.items():
            cooked = [precook(ref, n) for ref in refs]
            cooked_refs[vid] = cooked
            seen = set()
            for ref in cooked:
                seen.update(ref.keys())
            for ngram in seen:
                self.doc_freq[ngram] += 1
        self.log_num_docs = np.log(float(len(vid2refs)))

        # --- Precompute vector tf-idf của mọi reference (refs cố định) ---
        self.ref_vecs = {
            vid: [self._counts2vec(c) for c in cooked]
            for vid, cooked in cooked_refs.items()
        }
        self.num_refs = {vid: len(refs) for vid, refs in vid2refs.items()}

    def _counts2vec(self, cnts):
        """dict ngram->tf  ->  (vec tf-idf theo bậc n, norm L2 theo bậc, độ dài unigram)"""
        vec = [defaultdict(float) for _ in range(self.n)]
        norm = [0.0] * self.n
        length = 0
        for ngram, term_freq in cnts.items():
            df = np.log(max(1.0, self.doc_freq[ngram]))
            k = len(ngram) - 1
            vec[k][ngram] = float(term_freq) * (self.log_num_docs - df)
            norm[k] += vec[k][ngram] ** 2
            if k == 0:
                length += term_freq
        norm = [np.sqrt(x) for x in norm]
        return vec, norm, length

    def _sim(self, vec_h, norm_h, len_h, vec_r, norm_r, len_r):
        """CIDEr-D similarity: cosine có clip tf (min) + phạt lệch độ dài."""
        delta = float(len_h - len_r)
        val = np.zeros(self.n)
        for k in range(self.n):
            for ngram, count in vec_h[k].items():
                val[k] += min(count, vec_r[k][ngram]) * vec_r[k][ngram]
            if norm_h[k] != 0 and norm_r[k] != 0:
                val[k] /= norm_h[k] * norm_r[k]
        val *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
        return val

    def __call__(self, vids: List[str], hyps: List[str]) -> np.ndarray:
        """vids[i] là video của hyps[i] (một video xuất hiện K lần liên tiếp).
        Trả reward CIDEr-D (thang x10 như pycocoevalcap) shape (len(hyps),)."""
        rewards = np.zeros(len(hyps), dtype=np.float32)
        for i, (vid, hyp) in enumerate(zip(vids, hyps)):
            vec_h, norm_h, len_h = self._counts2vec(precook(hyp, self.n))
            score = np.zeros(self.n)
            for vec_r, norm_r, len_r in self.ref_vecs[vid]:
                score += self._sim(vec_h, norm_h, len_h, vec_r, norm_r, len_r)
            rewards[i] = np.mean(score) / self.num_refs[vid] * 10.0
        return rewards
