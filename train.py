from __future__ import print_function

import gc
import torch
import random
import time
import numpy as np
import json
import os
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from config import TrainConfig as C
from models.abd_transformer import ABDTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from utils import evaluate, load_checkpoint, save_checkpoint, test, train


def build_loaders():
    if   C.corpus == "MSVD":   corpus = MSVD(C)
    elif C.corpus == "MSRVTT": corpus = MSRVTT(C)
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        corpus.vocab.n_vocabs, corpus.vocab.n_vocabs_untrimmed, corpus.vocab.n_words,
        corpus.vocab.n_words_untrimmed, C.loader.min_count))
    return corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab


def build_model(vocab):
    model = ABDTransformer(
        vocab=vocab,
        d_feat=C.feat.feature_dims,
        d_model=C.transformer.d_model,
        d_ff=C.transformer.d_ff,
        n_heads=C.transformer.n_heads,
        n_heads_big=C.transformer.n_heads_big,
        n_enc_layers=C.transformer.n_enc_layers,
        n_dec_layers=C.transformer.n_dec_layers,
        dropout=C.transformer.dropout,
        max_caption_len=C.loader.max_caption_len,
    )
    model.cuda()
    return model


# refers: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def log_train(e, loss, reg_lambda, time_taken, lr, summary):
    print(f"[EPOCH {e} TRAIN]")
    print(f"time: {time_taken:.2f} seconds | lr: {lr:.6f}")
    print(f"loss:{loss['total']:.6f} = (1-reg):{1-reg_lambda:.2f} * r2l_loss:{loss['r2l_loss']:.6f} + (reg):{reg_lambda:.2f} * l2r_loss:{loss['l2r_loss']:.6f}")
    # Store summary
    summary[e] = {
        "total_loss": loss['total'],
        "r2l_loss": loss['r2l_loss'],
        "l2r_loss": loss['l2r_loss'],
        "time": time_taken,
        "lr": lr,
    }


def log_val(e, loss, reg_lambda, r2l_scores, l2r_scores, time_taken, summary):
    print(f"[EPOCH {e} VAL]")
    print(f"time: {time_taken:.2f} seconds")
    print(f"loss:{loss['total']:.6f} = (1-reg):{1-reg_lambda:.2f} * r2l_loss:{loss['r2l_loss']:.6f} + (reg):{reg_lambda:.2f} * l2r_loss:{loss['l2r_loss']:.6f}")
    print(f"r2l_scores: {r2l_scores}")
    print(f"l2r_scores: {l2r_scores}")
    # Store summary
    summary[e] = {
        "total_loss": loss['total'],
        "r2l_loss": loss['r2l_loss'],
        "l2r_loss": loss['l2r_loss'],
        "r2l_scores": r2l_scores,
        "l2r_scores": l2r_scores,
        "time": time_taken,
    }


def log_test(r2l_scores, l2r_scores, time_taken):
    print(f"time: {time_taken:.2f} seconds")
    print(f"r2l_scores: {r2l_scores}")
    print(f"l2r_scores: {l2r_scores}")
    # return summary
    summary = {
        "r2l_scores": r2l_scores,
        "l2r_scores": l2r_scores,
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

    train_iter, val_iter, test_iter, vocab = build_loaders()

    model = build_model(vocab)

    parameter_number = get_parameter_number(model)
    print(parameter_number)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=C.lr,
        weight_decay=C.weight_decay,
        amsgrad=True
    )
    # warmup scheduler (LinearLR): linearly increase from start_factor*lr -> lr in warmup_epochs steps
    warmup_sched = LinearLR(
        optimizer,
        end_factor=1.0,
        total_iters=C.warmup_epochs,
    )
    # ReduceLROnPlateau for after warmup
    plateau_sched = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=C.lr_decay_gamma,
        patience=C.lr_decay_patience
    )

    # Find the best model
    best_val_CIDEr: float = float("-inf")
    best_epoch: int = -1
    best_ckpt_fpath: str = ""
    # Time taken for training, validation, and testing
    total_train_time: float = 0.0
    total_val_time: float = 0.0
    # Logging summaries
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
            vocab=vocab,
            reg_lambda=C.reg_lambda,
            gradient_clip=C.gradient_clip
        )
        _train_end_time = time.time()
        _train_time_taken = _train_end_time - _train_start_time
        total_train_time += _train_time_taken
        
        log_train(e=e, loss=train_loss, reg_lambda=C.reg_lambda,
                  time_taken=_train_time_taken, lr=get_lr(optimizer), 
                  summary=train_summary)

        """ Validation """
        val_loss = test(
            model=model,
            val_iter=val_iter,
            vocab=vocab,
            reg_lambda=C.reg_lambda
        )
        
        _val_start_time = time.time()
        r2l_val_scores, l2r_val_scores = evaluate(
            data_iter=val_iter,
            model=model,
            vocab=vocab,
            beam_size=C.beam_size,
            max_len=C.loader.max_caption_len
        )
        _val_end_time = time.time()
        _val_time_taken = _val_end_time - _val_start_time
        total_val_time += _val_time_taken
        
        log_val(e=e, loss=val_loss, reg_lambda=C.reg_lambda,
                r2l_scores=r2l_val_scores, l2r_scores=l2r_val_scores,
                time_taken=_val_time_taken, summary=val_summary)

        """ Learning Rate Decay & Checkpointing """
        if e <= C.warmup_epochs:
            # print(f">> Epoch {e} in warmup phase, applying warmup scheduler.")
            warmup_sched.step()
        else:
            # print(f">> Epoch {e} in normal phase, applying plateau scheduler.")
            plateau_sched.step(val_loss['total'])
        """
        if e >= C.lr_decay_start_from:
            lr_scheduler.step(val_loss['total'])
        """;
        if l2r_val_scores['CIDEr'] > best_val_CIDEr:
            best_epoch = e
            best_val_CIDEr = l2r_val_scores['CIDEr']
            best_ckpt_fpath = ckpt_fpath

            print(f">> Saving checkpoint to {ckpt_fpath.split('/')[-1]}")
            save_checkpoint(model=model, ckpt_fpath=ckpt_fpath)
            print(f">> New best model at epoch {e} | CIDEr: {best_val_CIDEr}")

    """ Test with Best Model """
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n\n\n[BEST: {best_epoch} | SEED: {seed} | VAL-CIDEr: {best_val_CIDEr}]")
    best_model = load_checkpoint(model=model, ckpt_fpath=best_ckpt_fpath)
    
    _test_start_time = time.time()
    r2l_best_scores, l2r_best_scores = evaluate(
        data_iter=test_iter, 
        model=best_model, 
        vocab=vocab, 
        beam_size=C.beam_size, 
        max_len=C.loader.max_caption_len
    )
    _test_end_time = time.time()
    _test_time_taken = _test_end_time - _test_start_time
    
    test_summary = log_test(
        r2l_scores=r2l_best_scores, 
        l2r_scores=l2r_best_scores, 
        time_taken=_test_time_taken
    )
    
    save_log_summary(train_summary=train_summary,
                     val_summary=val_summary,
                     test_summary=test_summary,
                     log_folder=C.log_folder)
    
    print("-"*40)
    print(f">> [Train time] Total: {total_train_time:.2f} seconds => Per epoch: {total_train_time / C.epochs:.2f} seconds")
    print(f">> [Val time] Total: {total_val_time:.2f} seconds => Per epoch: {total_val_time / C.epochs:.2f} seconds")
    return


if __name__ == "__main__":
    main()
