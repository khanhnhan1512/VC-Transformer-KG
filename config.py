import os
from typing import List
import time


class FeatureConfig:
    # --- New single features (ưu tiên pooled/[CLS] trước, rồi mean) ---
    # model: str = "Blip2VitGPooledKF"       # EVA-ViT-g [CLS] token
    model: str = "SigLIP2GiantPooledKF"  # SigLIP2-giant pooler_output (MAP head)
    # model: str = "Blip2VitGMeanKF"       # EVA-ViT-g mean của patch token
    # model: str = "SigLIP2GiantMeanKF"    # SigLIP2-giant mean của patch token
    # --- Old features ---
    # model: str = "newBlip2ClsKF+newImgCapBlip2KF+newMViTv2"
    # model: str = "Blip2QFormerMeanKF"
    # model: str = "newBlip2ClsKF"
    feature_dims: List[int] = []

    for modality in model.split("+"):
        # Appearance feature dimension
        if   modality.find('newBlip2ClsKF') != -1:      feature_dims.append(1408)
        # Semantic feature dimension
        elif modality.find('newImgCapBlip2KF') != -1:   feature_dims.append(1024)
        # Motion feature dimension
        elif modality.find('newMViTv2') != -1:          feature_dims.append(768)
        # BLIP-2 Q-Former feature dimension (mean của 32 query token)
        elif modality.find('Blip2QFormerMeanKF') != -1: feature_dims.append(768)
        # BLIP-2 ViT-g feature dimension (CLS/Pooled hoặc Mean của patch token)
        elif modality.find('Blip2VitG') != -1:          feature_dims.append(1408)
        # SigLIP2-giant feature dimension (pooler_output/Pooled hoặc Mean của patch token)
        elif modality.find('SigLIP2Giant') != -1:       feature_dims.append(1536)


class VocabConfig:
    init_word2idx = {'<PAD>': 0, '<S>': 1}


class MSVDLoaderConfig:
    # Flexible to change the path to data folder when run on Kaggle
    DATA_FOLDER_PATH = "./data"
    if not os.path.exists(DATA_FOLDER_PATH): DATA_FOLDER_PATH = "/kaggle/input/datasets/vmphat/msvd-blip2qformer"
    # if not os.path.exists(DATA_FOLDER_PATH): DATA_FOLDER_PATH = "/kaggle/input/datasets/vmphat/bidect-msvd-dataset"

    # caption_fpath = "./data/MSVD/metadata/<FILENAME>.csv"
    train_caption_fpath = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/train.csv")
    val_caption_fpath   = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/val.csv")
    test_caption_fpath  = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/test.csv")

    # phase_video_feat_fpath_tpl = "./data/{}/features/{}_{}.hdf5"
    phase_video_feat_fpath_tpl = DATA_FOLDER_PATH + "/{}/features/{}_{}.hdf5"

    num_workers = 4
    frame_sample_len = 9 #8
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

    num_workers = 4
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

    num_workers = 4
    frame_sample_len = 9 #P75
    frame_sampling_method = 'uniform'
    assert frame_sampling_method in ['uniform', 'random']


class TransformerConfig:
    """
    Flan-T5 family (params | d_model | #decoder layer | #head | d_ff):
      small  ~80M  | d_model 512  | 8  layer | 6  head | d_ff 1024 (gated)
      base   ~250M | d_model 768  | 12 layer | 12 head | d_ff 2048 (gated)
      large  ~780M | d_model 1024 | 24 layer | 16 head | d_ff 2816 (gated)
      xl     ~3B   | d_model 2048 | 24 layer | 32 head | d_ff 5120 (gated)
      xxl    ~11B  | d_model 4096 | 24 layer | 64 head | d_ff 10240 (gated)
    d_model càng lớn -> projection từ feature (SigLIP2 1536-d) càng ít mất mát.
    """;
    # t5_model_name = "google/flan-t5-small"  #  80M params | d_model 512
    t5_model_name = "google/flan-t5-base"   # 250M params | d_model 768
    # t5_model_name = "google/flan-t5-large"  # 780M params | d_model 1024
    # t5_model_name = "google/flan-t5-xl"     #   3B params | d_model 2048

    dropout = 0.1
    max_caption_tokens = 32

    # Số block T5 decoder giữ lại (0 = giữ nguyên toàn bộ).
    # Chọn N block cách đều (linspace), luôn gồm block 0 (mang relative attention bias):
    #   small/base: 8/12 block  | large/xl/xxl: 24 block
    #   large giữ 6 -> block [0, 5, 9, 14, 18, 23]
    num_decoder_layers = 6


class TrainConfig:
    corpus = "MSVD"
    # corpus = "MSRVTT"
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
    weight_decay = 5e-6
    # Fine-tune pretrained Flan-T5 decoder nên không cần warmup (0 = tắt;
    # train from scratch mới cần)
    warmup_epochs = 0
    label_smoothing = 0.15
    beam_size = 5

    """ Evaluation Metrics """
    metrics = ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']

    """ ID """
    feat_id = f"FEAT {feat.model} "\
              f"fsl-{loader.frame_sample_len}"

    transformer_id = f"T5 "\
                     f"{transformer.t5_model_name} " \
                     f"mct-{transformer.max_caption_tokens} " \
                     f"dp-{transformer.dropout} " \
                     f"dl-{transformer.num_decoder_layers}"

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
