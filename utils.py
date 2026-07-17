# coding=utf-8
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict
from weakref import WeakKeyDictionary
from config import TrainConfig as C

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses
        self.losses = [[] for _ in range(self.num_losses)]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses
        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [0. for _ in range(self.num_losses)]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def parse_batch(batch):
    # non_blocking=True: copy H2D được XẾP HÀNG rồi trả về ngay, chồng lấn với
    # compute của batch trước. An toàn vì nguồn đã pinned (pin_memory=True trong
    # DataLoader) và mọi thao tác sau đó đều nằm trên CUDA stream nên đúng thứ tự.
    vids, feats_list, caption_ids, caption_mask, raw_captions = batch
    feats = tuple([f.cuda(non_blocking=True) for f in feats_list])
    caption_ids = caption_ids.cuda(non_blocking=True)
    caption_mask = caption_mask.cuda(non_blocking=True)
    return vids, feats, caption_ids, caption_mask, raw_captions


def train(e, model, optimizer, train_iter, tokenizer, gradient_clip):
    model.train()
    loss_checker = LossChecker(1)
    pad_token_id = tokenizer.pad_token_id
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=C.label_smoothing)

    t = tqdm(train_iter)
    for batch in t:
        _, feats, caption_ids, caption_mask, _ = parse_batch(batch)

        labels = caption_ids.clone()
        labels[labels == pad_token_id] = -100

        optimizer.zero_grad()
        outputs = model(feats, labels=labels, decoder_attention_mask=caption_mask)

        loss = loss_fct(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1)
        )
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_checker.update(loss.item())
        t.set_description(
            "[Epoch #{0}] loss:{1:.4f}".format(e, *loss_checker.mean(last=10))
        )

    total_loss = loss_checker.mean()
    return {'total': total_loss[0]}


def test(model, val_iter, tokenizer):
    model.eval()
    loss_checker = LossChecker(1)
    pad_token_id = tokenizer.pad_token_id
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=C.label_smoothing)

    t = tqdm(val_iter)
    t.set_description('Test:')
    with torch.no_grad():
        for batch in t:
            _, feats, caption_ids, caption_mask, _ = parse_batch(batch)

            labels = caption_ids.clone()
            labels[labels == pad_token_id] = -100

            outputs = model(feats, labels=labels, decoder_attention_mask=caption_mask)
            loss = loss_fct(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1)
            )
            loss_checker.update(loss.item())

    total_loss = loss_checker.mean()
    return {'total': total_loss[0]}


# Dữ liệu eval là TĨNH qua các epoch (chỉ trọng số model đổi), nên chỉ quét
# loader đúng MỘT lần rồi tái dùng. WeakKeyDictionary: cache tự biến mất khi
# loader được thu hồi, không giữ tham chiếu sống.
_eval_data_cache: "WeakKeyDictionary" = WeakKeyDictionary()


def _get_eval_data(data_iter) -> Tuple[List[str], List[tuple], Dict[str, list]]:
    """Quét loader 1 lần -> (vids, feats mỗi video, caption ground-truth). Có cache.

    Trước đây evaluate() quét loader HAI lần MỖI epoch: một để gom feature, một
    để gom GT caption. Cả hai đều không đổi giữa các epoch -> gộp thành một lượt
    duy nhất và cache lại.
    """
    cached = _eval_data_cache.get(data_iter)
    if cached is not None:
        return cached

    vids_order: List[str] = []
    feats_per_vid: List[tuple] = []
    vid2GTs: Dict[str, list] = defaultdict(list)
    seen_vids = set()

    for batch in tqdm(iter(data_iter), desc='Build eval cache'):
        vids, feats, _, _, raw_captions = parse_batch(batch)
        for i, vid in enumerate(vids):
            vid2GTs[vid].append(raw_captions[i])
            if vid in seen_vids:
                continue
            seen_vids.add(vid)
            vids_order.append(vid)
            # f[i:i+1] giữ nguyên chiều batch và MỌI chiều sau đó:
            #   (B,T,D) -> (1,T,D)            appearance pre-extracted
            #   (B,T,K,C,G,G) -> (1,T,K,C,G,G) motion vector grid (THÔ)
            feats_per_vid.append(tuple(f[i:i + 1] for f in feats))

    cached = (vids_order, feats_per_vid, vid2GTs)
    _eval_data_cache[data_iter] = cached
    return cached


def get_predicted_captions(data_iter, model, tokenizer, beam_size, max_len, batch_size):
    """Sinh caption cho mỗi video (1 lần/video), gom nhiều video vào một lần generate().

    Trước đây gọi generate() với batch=1 cho từng video -> GPU gần như rỗi và
    beam search không tận dụng được song song. Gom batch cho kết quả TƯƠNG ĐƯƠNG
    (mọi video cùng num_gop; GOP pad đã do attention mask xử lý) nhưng nhanh hơn
    nhiều lần.
    """
    vids_order, feats_per_vid, _ = _get_eval_data(data_iter)

    model.eval()
    vid2pred: Dict[str, str] = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(vids_order), batch_size), desc='Generate'):
            chunk_vids = vids_order[start:start + batch_size]
            chunk_feats = feats_per_vid[start:start + batch_size]
            # zip(*) gom theo modality, cat -> (B, T, ...) cho từng modality
            batched = tuple(torch.cat(mod, dim=0) for mod in zip(*chunk_feats))
            captions = model.generate_captions(batched, tokenizer, beam_size, max_len)
            vid2pred.update(zip(chunk_vids, captions))

    return vid2pred


def get_groundtruth_captions(data_iter):
    return _get_eval_data(data_iter)[2]


def score(vid2pred, vid2GTs):
    assert set(vid2pred.keys()) == set(vid2GTs.keys()), \
        f"[score] #vid2pred={len(vid2pred)} != #vid2GTs={len(vid2GTs)}"
    vid2idx = {v: i for i, v in enumerate(vid2pred.keys())}
    refs = {vid2idx[vid]: GTs for vid, GTs in vid2GTs.items()}
    hypos = {vid2idx[vid]: [pred] for vid, pred in vid2pred.items()}
    scores = calc_scores(refs, hypos)
    return scores


def calc_scores(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def evaluate(data_iter, model, tokenizer, beam_size, max_len, return_captions,
             eval_batch_size):
    vid2pred = get_predicted_captions(data_iter, model, tokenizer, beam_size, max_len,
                                      eval_batch_size)
    vid2GTs = get_groundtruth_captions(data_iter)
    scores = score(vid2pred, vid2GTs)
    if return_captions:
        return scores, vid2pred, vid2GTs
    else:
        return scores


def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.load_state_dict(checkpoint['t5_captioner'])
    return model


def save_checkpoint(model, ckpt_fpath):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    os.makedirs(ckpt_dpath, exist_ok=True)
    torch.save({'t5_captioner': model.state_dict()}, ckpt_fpath)
