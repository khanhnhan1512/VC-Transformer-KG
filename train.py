from __future__ import print_function

import gc
import torch
import random
import time
import numpy as np
import json
import os
from typing import Dict, List
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from loader.VATEX import VATEX
from config import TrainConfig as C
from models.t5_captioner import T5Captioner
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from utils import evaluate, load_checkpoint, save_checkpoint, test, train


def build_loaders():
    if   C.corpus == "MSVD"  : corpus = MSVD(C)
    elif C.corpus == "MSRVTT": corpus = MSRVTT(C)
    elif C.corpus == "VATEX" : corpus = VATEX(C)
    print('T5 tokenizer vocab size: {}'.format(corpus.tokenizer.vocab_size))
    return corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.tokenizer


def build_model():
    model = T5Captioner(
        d_feat=C.feat.feature_dims,
        t5_model_name=C.transformer.t5_model_name,
        dropout=C.transformer.dropout,
    )
    model.cuda()
    return model


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def log_train(e, loss, time_taken, lr, summary):
    print(f"[EPOCH {e} TRAIN]")
    print(f"time: {time_taken:.2f} seconds | lr: {lr:.6f}")
    print(f"loss: {loss['total']:.6f}")
    summary[e] = {
        "loss": loss['total'],
        "time": time_taken,
        "lr": lr,
    }


def log_val(e, loss, scores, time_taken, summary):
    print(f"[EPOCH {e} VAL]")
    print(f"time: {time_taken:.2f} seconds")
    print(f"loss: {loss['total']:.6f}")
    print(f"scores: {scores}")
    summary[e] = {
        "loss": loss['total'],
        "scores": scores,
        "time": time_taken,
    }


def log_test(scores, time_taken):
    print(f"time: {time_taken:.2f} seconds")
    print(f"scores: {scores}")
    summary = {
        "scores": scores,
        "time": time_taken,
    }
    return summary


def save_log_summary(train_summary, val_summary, test_summary, log_folder):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    with open(os.path.join(log_folder, 'train_summary.json'), 'w') as f:
        json.dump(train_summary, f, indent=4)

    with open(os.path.join(log_folder, 'val_summary.json'), 'w') as f:
        json.dump(val_summary, f, indent=4)

    with open(os.path.join(log_folder, 'test_summary.json'), 'w') as f:
        json.dump(test_summary, f, indent=4)


def save_test_qualitative_results(vid2pred: Dict, vid2GTs: Dict, log_folder: str) -> None:
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    results: List[Dict] = []
    vids = vid2pred.keys()
    for vid in vids:
        results.append({
            "videoID": vid,
            "predCap": vid2pred[vid],
            "gtCap": vid2GTs[vid]
        })

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    with open(os.path.join(log_folder, 'test_caption_comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main():
    print(f"MODEL ID: {C.model_id}")

    seed: int = 904666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_iter, val_iter, test_iter, tokenizer = build_loaders()

    model = build_model()

    parameter_number = get_parameter_number(model)
    print(parameter_number)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=C.lr,
        weight_decay=C.weight_decay,
        amsgrad=True
    )
    warmup_sched = LinearLR(
        optimizer,
        end_factor=1.0,
        total_iters=C.warmup_epochs,
    )
    plateau_sched = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=C.lr_decay_gamma,
        patience=C.lr_decay_patience
    )

    best_val_CIDEr: float = float("-inf")
    best_val_scores: Dict[str, float] = {
        "Bleu_4": float("-inf"), "METEOR": float("-inf"),
        "ROUGE_L": float("-inf"), "CIDEr": float("-inf")
    }
    best_epoch: int = -1
    best_ckpt_fpath: str = ""
    total_train_time: float = 0.0
    total_val_time: float = 0.0
    train_summary = {}
    val_summary = {}

    for e in range(1, C.epochs + 1):
        ckpt_fpath = C.ckpt_fpath_tpl.format(e)

        """ Train """
        print("\n")
        _train_start_time = time.time()
        train_loss = train(
            e=e,
            model=model,
            optimizer=optimizer,
            train_iter=train_iter,
            tokenizer=tokenizer,
            gradient_clip=C.gradient_clip
        )
        _train_end_time = time.time()
        _train_time_taken = _train_end_time - _train_start_time
        total_train_time += _train_time_taken

        log_train(e=e, loss=train_loss,
                  time_taken=_train_time_taken, lr=get_lr(optimizer),
                  summary=train_summary)

        """ Validation """
        val_loss = test(
            model=model,
            val_iter=val_iter,
            tokenizer=tokenizer
        )

        _val_start_time = time.time()
        val_scores = evaluate(
            data_iter=val_iter,
            model=model,
            tokenizer=tokenizer,
            beam_size=C.beam_size,
            max_len=C.transformer.max_caption_tokens,
            return_captions=False
        )
        _val_end_time = time.time()
        _val_time_taken = _val_end_time - _val_start_time
        total_val_time += _val_time_taken

        log_val(e=e, loss=val_loss, scores=val_scores,
                time_taken=_val_time_taken, summary=val_summary)

        """ Learning Rate Decay & Checkpointing """
        if e <= C.warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(val_loss['total'])

        n_better_metrics = 0
        if val_scores["Bleu_4"]  > best_val_scores["Bleu_4"] : n_better_metrics += 1
        if val_scores["METEOR"]  > best_val_scores["METEOR"] : n_better_metrics += 1
        if val_scores["ROUGE_L"] > best_val_scores["ROUGE_L"]: n_better_metrics += 1
        if val_scores["CIDEr"]   > best_val_scores["CIDEr"]  : n_better_metrics += 1

        if (val_scores['CIDEr'] > best_val_CIDEr) and (n_better_metrics >= 3):
            best_epoch = e
            best_val_CIDEr  = val_scores['CIDEr']
            best_val_scores = val_scores
            best_ckpt_fpath = ckpt_fpath

            save_checkpoint(model=model, ckpt_fpath=ckpt_fpath)
            print(f">> New best model at epoch {e} | N_Metrics: {n_better_metrics} | CIDEr: {best_val_CIDEr}")

    """ Test with Best Model """
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n\n\n[BEST: {best_epoch} | SEED: {seed} | VAL-CIDEr: {best_val_CIDEr}]")
    best_model = load_checkpoint(model=model, ckpt_fpath=best_ckpt_fpath)

    _test_start_time = time.time()
    test_scores, test_vid2pred, test_vid2GTs = evaluate(
        data_iter=test_iter,
        model=best_model,
        tokenizer=tokenizer,
        beam_size=C.beam_size,
        max_len=C.transformer.max_caption_tokens,
        return_captions=True
    )
    _test_end_time = time.time()
    _test_time_taken = _test_end_time - _test_start_time

    test_summary = log_test(
        scores=test_scores,
        time_taken=_test_time_taken
    )

    save_log_summary(train_summary=train_summary,
                     val_summary=val_summary,
                     test_summary=test_summary,
                     log_folder=C.log_folder)
    save_test_qualitative_results(vid2pred=test_vid2pred,
                                  vid2GTs=test_vid2GTs,
                                  log_folder=C.log_folder)

    print("-"*64)
    print(f">> [Train time] Total: {total_train_time:.2f} seconds => Per epoch: {total_train_time / C.epochs:.2f} seconds")
    print(f">> [Val time] Total: {total_val_time:.2f} seconds => Per epoch: {total_val_time / C.epochs:.2f} seconds")
    return


def print_gpu_info() -> None:
    if not torch.cuda.is_available():
        print("Không có GPU CUDA khả dụng.")
        return

    n = torch.cuda.device_count()
    print(f"Found {n} CUDA device(s)\n")

    for i in range(n):
        dev = torch.device(f"cuda:{i}")
        props = torch.cuda.get_device_properties(dev)
        print(f"Device {i}: {props.name}")
        print(f"  Total memory bytes: {props.total_memory}")
        print(f"  MultiProcessor count: {props.multi_processor_count}")
        print(f"  Major.Minor: {props.major}.{props.minor}")
        print(f"  Max threads per block: {props.max_threads_per_multi_processor}")
        print("=" * 64)


if __name__ == "__main__":
    print_gpu_info()
    main()
