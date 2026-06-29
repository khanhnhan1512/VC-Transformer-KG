# coding=utf-8
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict
from collections import defaultdict
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
    vids, feats_list, caption_ids, caption_mask, raw_captions = batch
    feats = tuple([f.cuda() for f in feats_list])
    caption_ids = caption_ids.cuda()
    caption_mask = caption_mask.cuda()
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


def get_predicted_captions(data_iter, model, tokenizer, beam_size, max_len):
    def build_onlyonce_iter(data_iter):
        seen_vids = set()
        onlyonce_iter = []

        for batch in tqdm(iter(data_iter)):
            vids, feats, _, _, _ = parse_batch(batch)
            for i, vid in enumerate(vids):
                if vid not in seen_vids:
                    seen_vids.add(vid)
                    feats_tup = tuple(f[i, :, :].unsqueeze(0) for f in feats)
                    onlyonce_iter.append((vid, feats_tup))

        return onlyonce_iter

    model.eval()
    onlyonce_iter = build_onlyonce_iter(data_iter)
    vid2pred: Dict[str, str] = {}

    with torch.no_grad():
        for vid, feats in tqdm(onlyonce_iter):
            captions = model.generate_captions(feats, tokenizer, beam_size, max_len)
            vid2pred[vid] = captions[0]

    return vid2pred


def get_groundtruth_captions(data_iter):
    vid2GTs = defaultdict(list)

    for batch in tqdm(iter(data_iter)):
        vids, _, _, _, raw_captions = parse_batch(batch)
        for vid, caption in zip(vids, raw_captions):
            vid2GTs[vid].append(caption)

    return vid2GTs


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


def evaluate(data_iter, model, tokenizer, beam_size, max_len, return_captions):
    vid2pred = get_predicted_captions(data_iter, model, tokenizer, beam_size, max_len)
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
