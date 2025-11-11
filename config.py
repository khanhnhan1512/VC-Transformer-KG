import os
from typing import List
import time


class FeatureConfig:
    model: str = "Blip2ClsKF+MViTv2+ImgCapBlip2KF"
    feature_dims: List[int] = []

    # Per-image feature dimension
    if   model.startswith('InceptionResNetV2+'): feature_dims.append(1536)
    # elif model.startswith('ResNet152+'):         feature_dims.append(2048)
    # elif model.startswith('BlipCls+'):           feature_dims.append(768)
    # elif model.startswith('BlipBaseClsKF+'):     feature_dims.append(768)
    # elif model.startswith('BlipBaseAvgKF+'):     feature_dims.append(768)
    elif model.startswith('Blip2ClsKF+'):        feature_dims.append(1408)

    # Motion feature dimension
    if   model.find('+I3D+') != -1:     feature_dims.append(1024)
    # elif model.find('+I3DRaft+') != -1: feature_dims.append(1024)
    elif model.find('+MViTv2+') != -1:  feature_dims.append(768)

    # Object feature dimension
    if   model.find('+OFeat') != -1:      feature_dims.append(1024)
    # elif model.find('+FasterRCNN') != -1: feature_dims.append(1028)
    # elif model.find('+MaskRCNNv2') != -1: feature_dims.append(1024)
    # elif model.find('+ImgCapKF') != -1:   feature_dims.append(384)
    # elif model.find('+ImgCapBlipLargeKF') != -1:   feature_dims.append(1024)
    elif model.find('+ImgCapBlip2KF') != -1:   feature_dims.append(1024)
    
    # Other feature dimension
    # if model.find('+rel') != -1:        feature_dims.append(300)


class VocabConfig:
    init_word2idx = {'<PAD>': 0, '<S>': 1}


class MSVDLoaderConfig:
    n_train = 1200
    n_val = 100
    n_test = 670

    # Flexible to change the path to data folder when run on Kaggle
    DATA_FOLDER_PATH = "./data"
    if not os.path.exists(DATA_FOLDER_PATH):
        DATA_FOLDER_PATH = "/kaggle/input/btkg-msvd-dataset"

    # caption_fpath = "./data/MSVD/metadata/<FILENAME>.csv"
    # total_caption_fpath = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/MSR Video Description Corpus.csv")
    train_caption_fpath = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/train.csv")
    val_caption_fpath = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/val.csv")
    test_caption_fpath = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/test.csv")

    min_count = 3
    max_caption_len = 30

    # total_video_feat_fpath_tpl = "./data/{}/features/{}.hdf5"
    # phase_video_feat_fpath_tpl = "./data/{}/features/{}_{}.hdf5"
    # total_video_feat_fpath_tpl = DATA_FOLDER_PATH + "/{}/features/{}.hdf5"
    phase_video_feat_fpath_tpl = DATA_FOLDER_PATH + "/{}/features/{}_{}.hdf5"

    frame_sampling_method = 'uniform'
    assert frame_sampling_method in ['uniform', 'random']
    frame_sample_len = 10

    num_workers = 4


class TransformerConfig:
    d_model = 512
    d_ff = 2048         # d_ff / d_model = 4
    n_heads_big = 128   # d_model / n_heads_big = 4
    n_heads = 8         # d_model / n_heads = 64
    n_enc_layers = 2    # Number of encoder layers
    n_dec_layers = 2    # Number of decoder layers
    dropout = 0.1


class TrainConfig:
    feat = FeatureConfig
    vocab = VocabConfig
    loader = MSVDLoaderConfig
    transformer = TransformerConfig

    """ Optimization """
    epochs = 25
    batch_size = 64
    gradient_clip = 5.0 # None if not used
    lr = 1e-4
    lr_decay_start_from = 3
    lr_decay_gamma = 0.5
    lr_decay_patience = 3
    weight_decay = 0.5e-5
    warmup_epochs = 3
    reg_lambda = 0.5    # weights of l2r
    beam_size = 4
    label_smoothing = 0.15

    """ Evaluation Metrics """
    metrics = ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']

    """ ID """
    feat_id = f"FEAT {feat.model} "\
              f"fsl-{loader.frame_sample_len} "\
              f"mcl-{loader.max_caption_len}"

    transformer_id = f"Transformer "\
                     f"d-{transformer.d_model} " \
                     f"N_enc-{transformer.n_enc_layers} " \
                     f"N_dec-{transformer.n_dec_layers} " \
                     f"h-{transformer.n_heads} " \
                     f"h_big-{transformer.n_heads_big} " \
                     f"dp-{transformer.dropout}"

    optimizer_id = f"OPTIM lr-{lr} warmup-{warmup_epochs} " \
                   f"gamma-{lr_decay_gamma} pat-{lr_decay_patience} " \
                   f"wd-{weight_decay} rg-{reg_lambda}"

    hyperparams_id = f"bs-{batch_size}-gc-{gradient_clip}-ls-{label_smoothing}"

    model_id = " = ".join(
        [feat_id, transformer_id, optimizer_id, hyperparams_id, str(int(time.time()))])

    """ Log """
    ckpt_dpath = f"./checkpoints/{model_id}"
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")


if __name__ == "__main__":
    C = TrainConfig()
    print(f">> Model ID:\n{C.model_id}")
