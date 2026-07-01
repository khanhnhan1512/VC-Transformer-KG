import os
from typing import List
import time


class FeatureConfig:
    model: str = "newBlip2ClsKF+newImgCapBlip2KF+newMViTv2"
    feature_dims: List[int] = []

    for modality in model.split("+"):
        # Appearance feature dimension
        if   modality.find('newBlip2ClsKF') != -1:    feature_dims.append(1408)
        # Semantic feature dimension
        elif modality.find('newImgCapBlip2KF') != -1: feature_dims.append(1024)
        # Motion feature dimension
        elif modality.find('newMViTv2') != -1:        feature_dims.append(768)


class VocabConfig:
    init_word2idx = {'<PAD>': 0, '<S>': 1}


class MSVDLoaderConfig:
    # Flexible to change the path to data folder when run on Kaggle
    DATA_FOLDER_PATH = "./data"
    if not os.path.exists(DATA_FOLDER_PATH): DATA_FOLDER_PATH = "/kaggle/input/datasets/vmphat/bidect-msvd-dataset"

    # caption_fpath = "./data/MSVD/metadata/<FILENAME>.csv"
    train_caption_fpath = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/train.csv")
    val_caption_fpath   = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/val.csv")
    test_caption_fpath  = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/test.csv")

    # phase_video_feat_fpath_tpl = "./data/{}/features/{}_{}.hdf5"
    phase_video_feat_fpath_tpl = DATA_FOLDER_PATH + "/{}/features/{}_{}.hdf5"

    min_count   = 3
    num_workers = 4
    max_caption_len  = 20
    frame_sample_len = 9 #P75
    frame_sampling_method = 'uniform'
    assert frame_sampling_method in ['uniform', 'random']


class MSRVTTLoaderConfig(object):
    # Flexible to change the path to data folder when run on Kaggle
    DATA_FOLDER_PATH = "./data"
    if not os.path.exists(DATA_FOLDER_PATH): DATA_FOLDER_PATH = "/kaggle/input/datasets/vmphat/bidect-msrvtt-dataset"

    train_caption_fpath = os.path.join(DATA_FOLDER_PATH, "MSRVTT/metadata/train.json")
    val_caption_fpath   = os.path.join(DATA_FOLDER_PATH, "MSRVTT/metadata/val.json")
    test_caption_fpath  = os.path.join(DATA_FOLDER_PATH, "MSRVTT/metadata/test.json")
    phase_video_feat_fpath_tpl = DATA_FOLDER_PATH + "/{}/features/{}_{}.hdf5"

    min_count   = 3
    num_workers = 4
    max_caption_len  = 20
    frame_sample_len = 13 #P75
    frame_sampling_method = 'uniform'
    assert frame_sampling_method in ['uniform', 'random']


class VATEXLoaderConfig(object):
    # Flexible to change the path to data folder when run on Kaggle
    DATA_FOLDER_PATH = "./data"
    if not os.path.exists(DATA_FOLDER_PATH): DATA_FOLDER_PATH = "/kaggle/input/datasets/vmphat/bidect-vatex-dataset"

    train_caption_fpath = os.path.join(DATA_FOLDER_PATH, "VATEX/metadata/vatex_train_english_v1.0_privacy_limited.json")
    val_caption_fpath   = os.path.join(DATA_FOLDER_PATH, "VATEX/metadata/vatex_val_english_v1.0_privacy_limited.json")
    test_caption_fpath  = os.path.join(DATA_FOLDER_PATH, "VATEX/metadata/vatex_public_test_english_v1.1_privacy_limited.json")
    phase_video_feat_fpath_tpl = DATA_FOLDER_PATH + "/{}/features/{}_{}.hdf5"

    min_count   = 3
    num_workers = 4
    max_caption_len  = 20
    frame_sample_len = 9 #P75
    frame_sampling_method = 'uniform'
    assert frame_sampling_method in ['uniform', 'random']


class TransformerConfig:
    t5_model_name = "google/flan-t5-small"  #  80M params
    # t5_model_name = "google/flan-t5-base"   # 250M params
    # t5_model_name = "google/flan-t5-large"  # 780M params

    dropout = 0.1
    max_caption_tokens = 32

    fusion_num_layers = 2
    fusion_n_heads = 8
    feat_mask_prob = 0.0
    num_decoder_layers = 0

    lora_r = 0
    lora_alpha = 16
    lora_target_modules = ['q', 'v']


class TrainConfig:
    # corpus = "MSVD"
    corpus = "MSRVTT"
    # corpus = "VATEX"
    if   corpus == "MSVD"  : loader = MSVDLoaderConfig
    elif corpus == "MSRVTT": loader = MSRVTTLoaderConfig
    elif corpus == "VATEX" : loader = VATEXLoaderConfig
    else: raise ValueError(f"Unknown corpus: {corpus}")

    feat        = FeatureConfig
    vocab       = VocabConfig
    transformer = TransformerConfig

    """ Optimization """
    epochs = 20
    batch_size = 64
    gradient_clip = 5.0 # None if not used
    lr = 1e-4
    lr_decay_gamma = 0.5
    lr_decay_patience = 3
    weight_decay = 0.5e-5
    warmup_epochs = 3
    label_smoothing = 0.15
    beam_size = 4

    """ Evaluation Metrics """
    metrics = ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']

    """ ID """
    feat_id = f"FEAT {feat.model} "\
              f"fsl-{loader.frame_sample_len} "\
              f"mcl-{loader.max_caption_len}"

    transformer_id = f"T5 "\
                     f"{transformer.t5_model_name} " \
                     f"mct-{transformer.max_caption_tokens} " \
                     f"dp-{transformer.dropout} " \
                     f"lora-r{transformer.lora_r}-a{transformer.lora_alpha}"

    optimizer_id = f"OPTIM lr-{lr} warmup-{warmup_epochs} " \
                   f"gamma-{lr_decay_gamma} pat-{lr_decay_patience} " \
                   f"wd-{weight_decay}"

    hyperparams_id = f"ep-{epochs} bs-{batch_size} gc-{gradient_clip} " \
                     f"bms-{beam_size} ls-{label_smoothing}"

    model_id = " = ".join(
        [corpus, feat_id, transformer_id, optimizer_id, hyperparams_id,
         str(int(time.time()))])

    """ Log """
    ckpt_dpath = f"./checkpoints/{model_id}"
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")
    log_folder = f"./logs/{model_id}"


if __name__ == "__main__":
    C = TrainConfig()
    print(f"Model ID:\n{C.model_id}")
    print(f"Feature dimensions: {C.feat.feature_dims}")
