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

    """ End-to-end pipeline (keyframes -> CLIP ViT) """
    # Folder chứa video .avi đã preprocess (tên file khớp format vid: {VideoID}_{Start}_{End}.avi)
    # Flexible to change the path to video folder when run on Kaggle
    VIDEO_FOLDER_PATH = "/mnt/d/___Video-Preprocessing/msvd/videos_240_h264_keyint_60"
    if not os.path.exists(VIDEO_FOLDER_PATH):
        VIDEO_FOLDER_PATH = "/kaggle/input/datasets/vmphat/msvd-videos-240-h264-keyint-60/msvd/videos_240_h264_keyint_60"
    # Số keyframe chuẩn hóa mỗi video: thiếu -> zero-pad (kèm mask), thừa -> uniform sampling trên tập keyframe
    # P75 của phân bố keyframe thực tế (video h264 keyint=60: min=1 max=33 mean=5.74, P50=5 P75=7 P90=10)
    keyframe_threshold = 7 #P75
    # Folder lưu disk cache keyframe đã trích (.npz) để khỏi decode lại video; để trống nếu không dùng
    frame_cache_dpath = ""

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
    # "e2e": keyframes -> CLIP ViT -> T5 decoder | "feats": pre-extracted features (đường cũ)
    pipeline = "e2e"
    assert pipeline in ["e2e", "feats"]

    clip_model_name = "openai/clip-vit-base-patch32"  # nhỏ nhất, để test code
    token_mode = "cls"            # hiện chỉ hỗ trợ "cls"; giữ field để mở rộng sau
    freeze_vision_encoder = False # False = fine-tune cả vision encoder
    # Chọn N layer cách đều (linspace, luôn gồm layer 0) của CLIP vision encoder
    # (0 = giữ nguyên toàn bộ, ViT-B có 12 layer)
    num_vision_layers = 4

    t5_model_name = "google/flan-t5-small"  #  80M params
    # t5_model_name = "google/flan-t5-base"   # 250M params
    # t5_model_name = "google/flan-t5-large"  # 780M params

    dropout = 0.1
    max_caption_tokens = 32

    fusion_num_layers = 2
    fusion_n_heads = 12
    feat_mask_prob = 0.0
    # Số layer T5 decoder (0 = giữ nguyên; flan-t5-small có 8 layer)
    # Pipeline e2e: chọn N layer cách đều (linspace, luôn gồm block 0 mang relative bias)
    # Pipeline feats: giữ N layer đầu như cũ
    num_decoder_layers = 4

    lora_r = 0
    lora_alpha = 16
    lora_target_modules = ['q', 'v']


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
    epochs = 15
    if transformer.pipeline == "e2e":
        # 64 caption-pair x 9 keyframe = 576 ảnh/step qua ViT -> OOM trên P100/T4
        batch_size = 32
        # Fine-tune full pretrained (CLIP + T5): lr 1e-4 quá cao, dễ phá pretrained weights
        lr = 3e-5
        # Mixed precision: tăng tốc đáng kể phần encoder ViT trên T4/P100
        use_amp = True
        # Dùng pretrained weights nên không cần warmup (0 = tắt; train from scratch mới cần)
        warmup_epochs = 0
    else:
        batch_size = 64
        lr = 1e-4
        use_amp = False  # giữ nguyên hành vi các run cũ để so sánh được
        warmup_epochs = 3
    gradient_clip = 5.0 # None if not used
    lr_decay_gamma = 0.5
    lr_decay_patience = 3
    weight_decay = 5e-5
    label_smoothing = 0.15
    beam_size = 5

    """ Evaluation Metrics """
    metrics = ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']

    if transformer.pipeline == "e2e" and not hasattr(loader, "keyframe_threshold"):
        raise ValueError(f"Pipeline 'e2e' hiện chỉ hỗ trợ MSVD (corpus={corpus} chưa có "
                         f"VIDEO_FOLDER_PATH/keyframe_threshold trong loader config)")

    """ ID """
    if transformer.pipeline == "e2e":
        feat_id = f"E2E {transformer.clip_model_name.split('/')[-1]} "\
                  f"tok-{transformer.token_mode} "\
                  f"kft-{loader.keyframe_threshold} "\
                  f"vl-{transformer.num_vision_layers} "\
                  f"dl-{transformer.num_decoder_layers} "\
                  f"mcl-{loader.max_caption_len}"
    else:
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
                     f"bms-{beam_size} ls-{label_smoothing} amp-{int(use_amp)}"

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
