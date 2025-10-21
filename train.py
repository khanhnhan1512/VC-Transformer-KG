from __future__ import print_function

import gc
import torch
import random
import numpy as np
from loader.MSVD import MSVD
from config import TrainConfig as C
from models.abd_transformer import ABDTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import evaluate, load_checkpoint, save_checkpoint, test, train


def build_loaders():
    corpus = MSVD(C)
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
        n_layers=C.transformer.n_layers,
        dropout=C.transformer.dropout,
        n_heads_big=C.transformer.n_heads_big
    )
    model.cuda()
    return model


def log_train(e, loss, reg_lambda, scores=None):
    print(f"[EPOCH {e} TRAIN]")
    print(f"loss:{loss['total']:.6f} = (1-reg):{1-reg_lambda:.2f} * r2l_loss:{loss['r2l_loss']:.6f} + (reg):{reg_lambda:.2f} * l2r_loss:{loss['l2r_loss']:.6f}")
    if scores is not None:
        print(f"scores: {scores}")


def log_val(e, loss, reg_lambda, r2l_scores, l2r_scores):
    print(f"[EPOCH {e} VAL]")
    print(f"loss:{loss['total']:.6f} = (1-reg):{1-reg_lambda:.2f} * r2l_loss:{loss['r2l_loss']:.6f} + (reg):{reg_lambda:.2f} * l2r_loss:{loss['l2r_loss']:.6f}")
    print(f"r2l_scores: {r2l_scores}")
    print(f"l2r_scores: {l2r_scores}")


def log_test(e, r2l_scores, l2r_scores):
    print(f"[EPOCH {e} TEST]")
    print(f"r2l_scores: {r2l_scores}")
    print(f"l2r_scores: {l2r_scores}")


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
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=C.lr_decay_gamma,
        patience=C.lr_decay_patience
    )

    best_val_CIDEr: float = float("-inf")
    best_epoch: int = -1
    best_ckpt_fpath: str = ""
    for e in range(1, C.epochs + 1):
        ckpt_fpath = C.ckpt_fpath_tpl.format(e)

        """ Train """
        print("\n")
        train_loss = train(
            e=e,
            model=model,
            optimizer=optimizer,
            train_iter=train_iter,
            vocab=vocab,
            reg_lambda=C.reg_lambda,
            gradient_clip=C.gradient_clip
        )
        log_train(e=e, loss=train_loss, reg_lambda=C.reg_lambda)

        """ Validation """
        val_loss = test(
            model=model,
            val_iter=val_iter,
            vocab=vocab,
            reg_lambda=C.reg_lambda
        )
        r2l_val_scores, l2r_val_scores = evaluate(
            data_iter=val_iter,
            model=model,
            vocab=vocab,
            beam_size=C.beam_size,
            max_len=C.loader.max_caption_len
        )
        log_val(e=e, loss=val_loss, reg_lambda=C.reg_lambda,
                r2l_scores=r2l_val_scores, l2r_scores=l2r_val_scores)

        """ Learning Rate Decay & Checkpointing """
        if e >= C.lr_decay_start_from:
            lr_scheduler.step(val_loss['total'])
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
    r2l_best_scores, l2r_best_scores = evaluate(
        data_iter=test_iter, 
        model=best_model, 
        vocab=vocab, 
        beam_size=C.beam_size, 
        max_len=C.loader.max_caption_len
    )
    print(f"r2l scores: {r2l_best_scores}")
    print(f"l2r scores: {l2r_best_scores}")
    print(">> Finish training!")
    
    return


if __name__ == "__main__":
    main()
