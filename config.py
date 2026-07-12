import os
from typing import List
import time


class FeatureConfig:
    # model: str = "newBlip2ClsKF+newImgCapBlip2KF+newMViTv2"
    model: str = "Blip2QFormerMeanKF"
    feature_dims: List[int] = []

    for modality in model.split("+"):
        # Appearance feature dimension
        if   modality.find('newBlip2ClsKF') != -1:      feature_dims.append(1408)
        # Semantic feature dimension
        elif modality.find('newImgCapBlip2KF') != -1:   feature_dims.append(1024)
        # Motion feature dimension
        elif modality.find('newMViTv2') != -1:          feature_dims.append(768)
        # BLIP-2 feature dimension
        elif modality.find('Blip2QFormerMeanKF') != -1: feature_dims.append(768)


class VocabConfig:
    init_word2idx = {'<PAD>': 0, '<S>': 1}


class MSVDLoaderConfig:
    # Flexible to change the path to data folder when run on Kaggle
    DATA_FOLDER_PATH = "./data"
    if not os.path.exists(DATA_FOLDER_PATH): DATA_FOLDER_PATH = "/kaggle/input/datasets/vmphat/bidect-msvd-dataset"
    if not os.path.exists(DATA_FOLDER_PATH): DATA_FOLDER_PATH = "/kaggle/input/datasets/vmphat/msvd-blip2qformer"

    # caption_fpath = "./data/MSVD/metadata/<FILENAME>.csv"
    train_caption_fpath = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/train.csv")
    val_caption_fpath   = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/val.csv")
    test_caption_fpath  = os.path.join(DATA_FOLDER_PATH, "MSVD/metadata/test.csv")

    # phase_video_feat_fpath_tpl = "./data/{}/features/{}_{}.hdf5"
    phase_video_feat_fpath_tpl = DATA_FOLDER_PATH + "/{}/features/{}_{}.hdf5"

    """ GOP motion (MV maps trích từ video cho motion encoder built-in) """
    # Folder chứa video .avi đã preprocess (tên file khớp vid: {VideoID}_{Start}_{End}.avi)
    VIDEO_FOLDER_PATH = "/mnt/d/___Video-Preprocessing/msvd/videos_240_h264_keyint_60"
    if not os.path.exists(VIDEO_FOLDER_PATH):
        VIDEO_FOLDER_PATH = "/kaggle/input/datasets/vmphat/msvd-videos-240-h264-keyint-60/msvd/videos_240_h264_keyint_60"
    # Folder cache MV maps đã trích (.npy) để khỏi decode lại video; để trống nếu không dùng
    mv_cache_dpath = ""

    num_workers = 4
    # Số GOP chuẩn hóa mỗi video — DÙNG CHUNG cho MỌI feature (BLIP-2, motion,
    # và các feature bổ sung sau này): thiếu -> zero-pad chung, thừa -> cùng bộ
    # chỉ số linspace (giữ alignment 1-1 giữa các modality)
    num_gop = 10


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
    t5_model_name = "google/flan-t5-small"  #  80M params
    # t5_model_name = "google/flan-t5-base"   # 250M params
    # t5_model_name = "google/flan-t5-large"  # 780M params

    dropout = 0.1
    max_caption_tokens = 32

    # Số block T5 decoder giữ lại (0 = giữ nguyên; flan-t5-small có 8 block)
    # Chọn N block cách đều (linspace), luôn gồm block 0 (mang relative attention bias)
    num_decoder_layers = 4

    """ GOP motion tokens (compressed-domain motion vectors) """
    # Mỗi GOP: token feature (BLIP-2, projection) + token motion (MotionEncoder
    # built-in trên MV map của các P/B-frame, train từ đầu)
    use_motion_tokens = True
    motion_grid_size = 16         # độ phân giải lưới MV map (2 x g x g)
    # Positional encoding cho chuỗi GOP: "index" (thứ tự) | "timestamp" (giây thật của I-frame)
    pos_encoding_type = "timestamp"
    assert pos_encoding_type in ["index", "timestamp"]


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
    weight_decay = 5e-5
    # Fine-tune pretrained Flan-T5 decoder nên không cần warmup (0 = tắt;
    # train from scratch mới cần)
    warmup_epochs = 0
    label_smoothing = 0.15
    beam_size = 5

    """ Evaluation Metrics """
    metrics = ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']

    """ ID """
    _seq_len = getattr(loader, 'num_gop', None) or getattr(loader, 'frame_sample_len', None)
    feat_id = f"FEAT {feat.model} "\
              f"gop-{_seq_len} "\
              f"mot-{int(transformer.use_motion_tokens)} "\
              f"pe-{transformer.pos_encoding_type}"

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
