# coding=utf-8
"""SCST — giai đoạn 2 tinh chỉnh model XE tốt nhất bằng policy gradient trên CIDEr-D.

Thuật toán (biến thể Luo 2020 "A Better Variant of SCST"):
  1. Với mỗi video: sample K caption bằng multinomial sampling (model hiện tại).
  2. Reward mỗi caption = CIDEr-D so với refs train (df toàn corpus, xem scst.py).
  3. Baseline = mean reward của K mẫu CÙNG video (không cần lượt greedy riêng).
  4. Loss = -(reward - baseline) * log p(chuỗi sample), chuẩn hóa theo số token.

HAI CÁCH CHẠY:
  - Pipeline 1 lệnh (khuyên dùng trên Kaggle): `python train.py` với
    `SCSTConfig.enabled = True` — XE xong tự chạy tiếp SCST trên best ckpt.
  - Độc lập trên checkpoint đã có (config phải GIỮ NGUYÊN như lần XE —
    kiến trúc lệch thì load_state_dict nổ, feature lệch thì im lặng vô nghĩa):
        python train_scst.py --ckpt "./checkpoints/<XE model id>/<e>.ckpt"
"""
import argparse
import os
import random
import time

import numpy as np
import torch

from config import TrainConfig as C
from scst import CiderDReward
from loader.data_loader_fusion import CustomDataset
from utils import evaluate, load_checkpoint, save_checkpoint, _get_eval_data


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _run_eval(data_iter, model, tokenizer):
    return evaluate(
        data_iter=data_iter, model=model, tokenizer=tokenizer,
        beam_size=C.beam_size, max_len=C.transformer.max_caption_tokens,
        return_captions=False, eval_batch_size=C.eval_batch_size,
    )


def run_scst(model, train_iter, val_iter, test_iter, tokenizer,
             out_dpath, epochs):
    """Giai đoạn SCST trên `model` (đã mang trọng số XE tốt nhất).

    Trả (best_val_cider, best_ckpt_fpath, test_scores). Luôn giữ best-val ckpt
    — nếu RL không cải thiện được epoch nào thì test bằng đúng trọng số XE
    ban đầu (snapshot scst_0.ckpt), không bao giờ tệ đi vì chạy SCST.
    """
    os.makedirs(out_dpath, exist_ok=True)
    print("\n" + "=" * 64)
    print(f"SCST | out: {out_dpath}")
    print(f"     | K={C.scst.num_samples} bs={C.scst.batch_size} video/bước "
          f"lr={C.scst.lr} ep={epochs}")
    print("=" * 64)

    # Feature + refs mức VIDEO của tập train (quét 1 lần, cache — xem utils)
    vids, feats_per_vid, vid2refs = _get_eval_data(train_iter)
    print(f"Train: {len(vids)} video, xây reward CIDEr-D (df toàn tập train)...")
    reward_fn = CiderDReward({v: vid2refs[v] for v in vids})

    # RL không cần weight decay; lr nhỏ — SCST chỉ "nắn" model, không học lại
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.scst.lr, weight_decay=0.0)

    # Snapshot epoch-0 = trọng số XE: mốc so sánh và fallback nếu SCST không ăn
    best_ckpt_fpath = os.path.join(out_dpath, "scst_0.ckpt")
    save_checkpoint(model, best_ckpt_fpath)
    val_scores = _run_eval(val_iter, model, tokenizer)
    best_val_cider = val_scores["CIDEr"]
    print(f"[SCST epoch 0 = XE] val CIDEr: {best_val_cider:.4f}")

    K = C.scst.num_samples
    bs = C.scst.batch_size
    max_len = C.transformer.max_caption_tokens
    rng = np.random.default_rng(42)
    total_train_time = 0.0

    for e in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(vids))
        sum_reward, sum_loss, n_steps = 0.0, 0.0, 0

        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            chunk_vids = [vids[i] for i in idx]
            # Gom theo modality -> (B, T, ...) như get_predicted_captions
            src = tuple(torch.cat(mod, dim=0)
                        for mod in zip(*[feats_per_vid[i] for i in idx]))

            # Encoder 1 lần: sampling không giữ graph (generate no_grad nội bộ),
            # scoring pass bên dưới tái dùng graph để gradient về encoder
            enc, enc_mask = model.encode_with_mask(src)
            seqs = model.sample_captions(enc, enc_mask, K, max_len)  # (B*K, L)

            hyps = tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # Cùng preprocess với refs của dataset -> so n-gram công bằng
            hyps = [CustomDataset.preprocess_caption(h) for h in hyps]
            rewards = reward_fn([v for v in chunk_vids for _ in range(K)], hyps)

            r = torch.from_numpy(rewards).to(enc.device).view(-1, K)
            adv = (r - r.mean(dim=1, keepdim=True)).reshape(-1, 1)  # (B*K, 1)

            logp, token_mask = model.sequence_logprobs(
                enc.repeat_interleave(K, dim=0),
                enc_mask.repeat_interleave(K, dim=0),
                seqs,
            )
            loss = -(adv * logp * token_mask).sum() / token_mask.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            if C.scst.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.scst.gradient_clip)
            optimizer.step()

            sum_reward += float(r.mean())
            sum_loss += float(loss)
            n_steps += 1

        t_epoch = time.time() - t0
        total_train_time += t_epoch
        print(f"[SCST epoch {e}] reward(≈train CIDEr-D): {sum_reward / n_steps:.4f} "
              f"| loss: {sum_loss / n_steps:.4f} | {t_epoch:.1f}s")

        val_scores = _run_eval(val_iter, model, tokenizer)
        print(f"[SCST epoch {e}] val scores: {val_scores}")
        if val_scores["CIDEr"] > best_val_cider:
            best_val_cider = val_scores["CIDEr"]
            best_ckpt_fpath = os.path.join(out_dpath, f"scst_{e}.ckpt")
            save_checkpoint(model, best_ckpt_fpath)
            print(f">> New best val CIDEr: {best_val_cider:.4f} -> {best_ckpt_fpath}")

    print(f"\n[SCST BEST] val CIDEr: {best_val_cider:.4f} | ckpt: {best_ckpt_fpath}")
    print(f">> [SCST train time] Total: {total_train_time:.2f}s "
          f"=> Per epoch: {total_train_time / max(epochs, 1):.2f}s")
    load_checkpoint(model, best_ckpt_fpath)
    t0 = time.time()
    test_scores = _run_eval(test_iter, model, tokenizer)
    print(f"[TEST sau SCST] time: {time.time() - t0:.1f}s\nscores: {test_scores}")
    return best_val_cider, best_ckpt_fpath, test_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="checkpoint XE tốt nhất")
    parser.add_argument("--out", default=None,
                        help="thư mục lưu checkpoint SCST (mặc định: <ckpt dir>-scst)")
    parser.add_argument("--epochs", type=int, default=C.scst.epochs)
    args = parser.parse_args()

    # Import trễ để tránh circular import (train.py cũng import run_scst từ đây)
    from train import build_loaders, build_model

    set_seed(42)
    train_iter, val_iter, test_iter, tokenizer = build_loaders()
    model = build_model()
    load_checkpoint(model, args.ckpt)
    print(f"SCST standalone | XE ckpt: {args.ckpt}")

    out_dpath = args.out or (os.path.dirname(os.path.abspath(args.ckpt)) + "-scst")
    run_scst(model, train_iter, val_iter, test_iter, tokenizer,
             out_dpath=out_dpath, epochs=args.epochs)


if __name__ == "__main__":
    main()
