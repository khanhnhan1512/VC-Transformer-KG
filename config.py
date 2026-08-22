import os
from typing import List
import time


class FeatureConfig:
    # `feature_spec`: các feature dùng cho model, nối bằng '+'. Mỗi tên khớp với
    # file HDF5 `{CORPUS}_{tên}_{phase}.hdf5`, và sinh ra 1 token cho mỗi GOP
    # (interleave: GOP thứ t -> [feat0_t, feat1_t, ...]).

    # --- New features (ưu tiên pooled/[CLS] trước, rồi mean) ---
    # feature_spec: str = "Blip2VitGPooledKF"     # EVA-ViT-g [CLS] token
    # feature_spec: str = "SigLIP2GiantPooledKF"  # SigLIP2-giant pooler_output (MAP head)
    # feature_spec: str = "Blip2VitGMeanKF"       # EVA-ViT-g mean của patch token
    # feature_spec: str = "SigLIP2GiantMeanKF"    # SigLIP2-giant mean của patch token

    # feature_spec: str = "SigLIP2GiantPooledKF+Blip2QFormerMeanKF"
    # feature_spec: str = "SigLIP2GiantPooledKF+SigLIP2GiantMeanKF"
    # feature_spec: str = "SigLIP2GiantPooledKF+MotionMV"   # grounded=True đã bật sẵn
    # feature_spec: str = "SigLIP2GiantPooledKF+newImgCapBlip2KF"

    # --- Old features ---
    feature_spec: str = "newBlip2ClsKF+newImgCapBlip2KF+newMViTv2"
    # feature_spec: str = "Blip2QFormerMeanKF"
    # feature_spec: str = "newBlip2ClsKF"

    # --- Feature THÔ ---
    # File HDF5 lưu dữ liệu CHƯA qua encoder nào (motion vector grid, shape
    # (NUM_GOP, K, C, G, G)), nên cần một encoder HỌC ĐƯỢC nằm trong model —
    # khác các feature pre-extracted vốn đã sẵn sàng để project thẳng.
    # `d_out` ở đây là HYPERPARAMETER (chiều output của encoder), KHÔNG phải
    # thuộc tính của file. Sau encoder, nó trở thành "một modality như mọi
    # modality khác" -> projection + type/positional embedding + interleave
    # dùng chung đường.
    raw_feature_cfgs = {
        "MotionMV": dict(
            d_out=512,       # chiều token motion; phải chia hết cho 8 (GroupNorm)
            in_channels=4,   # dx, dy, |v|, density
            num_bins=8,      # K đã lưu trong file HDF5
            grid_size=16,
            pool_bins=8,     # <= num_bins; đặt 4/2/1 để ablate K (pool có trọng số density)
            # grounded=True: FiLM — appearance token CÙNG GOP (feature pre-extracted
            # đầu tiên trong feature_spec, context_dim tự suy) điều biến các kênh
            # conv, để encoder biết "cái gì đang chuyển động" ngay lúc encode.
            # False = bản "mù" cũ (đã đo: 1.3124, trong nhiễu) — giữ làm ablation.
            grounded=True,
        ),
    }

    feature_names: List[str] = feature_spec.split("+")
    feature_dims: List[int] = []

    for modality in feature_names:
        # Feature THÔ -> dim = output của encoder tương ứng
        if   modality in raw_feature_cfgs:              feature_dims.append(raw_feature_cfgs[modality]["d_out"])
        # Appearance feature dimension
        elif modality.find('newBlip2ClsKF') != -1:      feature_dims.append(1408)
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
        else: raise ValueError(f"Unknown modality: {modality}")

    assert len(feature_dims) == len(feature_names)

    # Nhãn cấu hình của các feature THÔ đang dùng (vào model_id để các run
    # grounded/blind/pool_bins khác nhau không bị lẫn thư mục checkpoint/log).
    # Dùng vòng lặp thường, KHÔNG dùng genexpr: genexpr trong thân class có scope
    # riêng, không nhìn thấy biến class-level (raw_feature_cfgs) -> NameError.
    _raw_tags: List[str] = []
    for _n in feature_names:
        if _n in raw_feature_cfgs:
            _raw_tags.append(
                f"{_n}[{'grd' if raw_feature_cfgs[_n].get('grounded') else 'blind'}"
                f"-K{raw_feature_cfgs[_n]['pool_bins']}]")
    raw_id: str = " ".join(_raw_tags)


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

    num_workers = 4
    num_gop = 8  # P75


class MSRVTTLoaderConfig(object):
    # Flexible to change the path to data folder when run on Kaggle
    DATA_FOLDER_PATH = "./data"
    if not os.path.exists(DATA_FOLDER_PATH): DATA_FOLDER_PATH = "/kaggle/input/datasets/vmphat/bidect-msrvtt-dataset"

    train_caption_fpath = os.path.join(DATA_FOLDER_PATH, "MSRVTT/metadata/train.json")
    val_caption_fpath   = os.path.join(DATA_FOLDER_PATH, "MSRVTT/metadata/val.json")
    test_caption_fpath  = os.path.join(DATA_FOLDER_PATH, "MSRVTT/metadata/test.json")
    phase_video_feat_fpath_tpl = DATA_FOLDER_PATH + "/{}/features/{}_{}.hdf5"

    num_workers = 4
    num_gop = 6  # P75


class VATEXLoaderConfig(object):
    # Flexible to change the path to data folder when run on Kaggle
    DATA_FOLDER_PATH = "./data"
    if not os.path.exists(DATA_FOLDER_PATH): DATA_FOLDER_PATH = "/kaggle/input/datasets/vmphat/bidect-vatex-dataset"

    train_caption_fpath = os.path.join(DATA_FOLDER_PATH, "VATEX/metadata/vatex_train_english_v1.0_privacy_limited.json")
    val_caption_fpath   = os.path.join(DATA_FOLDER_PATH, "VATEX/metadata/vatex_val_english_v1.0_privacy_limited.json")
    test_caption_fpath  = os.path.join(DATA_FOLDER_PATH, "VATEX/metadata/vatex_public_test_english_v1.1_privacy_limited.json")
    phase_video_feat_fpath_tpl = DATA_FOLDER_PATH + "/{}/features/{}_{}.hdf5"

    num_workers = 4
    num_gop = 8  # P75


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
    t5_model_name = "google/flan-t5-small"  #  80M params | d_model 512
    # t5_model_name = "google/flan-t5-base"   # 250M params | d_model 768
    # t5_model_name = "google/flan-t5-large"  # 780M params | d_model 1024
    # t5_model_name = "google/flan-t5-xl"     #   3B params | d_model 2048

    dropout = 0.1
    max_caption_tokens = 32

    # Số block T5 decoder giữ lại (0 = giữ nguyên toàn bộ).
    # Chọn N block cách đều (linspace), luôn gồm block 0 (mang relative attention bias):
    #   small/base: 8/12 block  | large/xl/xxl: 24 block
    #   large giữ 6 -> block [0, 5, 9, 14, 18, 23]
    num_decoder_layers = 3

    # --- Generation / decode ---
    # CHỈ ảnh hưởng lúc generate (val + test); KHÔNG đổi trọng số. Nhưng val CIDEr
    # đổi -> có thể đổi epoch best-val được chọn, nên đây là knob thí nghiệm hợp lệ.
    # Giá trị trung tính (0 / 1.0 / 0) = HỆT hành vi cũ.
    #   no_repeat_ngram_size: chặn lặp n-gram, diệt caption vỡ kiểu
    #     "a cat is scratching a cat" / "a girl is putting on a girl" (0 = tắt).
    #   length_penalty: >1 ưu tiên câu DÀI hơn khi beam search (1.0 = trung tính).
    #     LƯU Ý CIDEr-D có phạt lệch độ dài -> dài hơn CHƯA CHẮC CIDEr cao hơn.
    #   min_new_tokens: ép độ dài sinh tối thiểu (0 = tắt).
    no_repeat_ngram_size = 3
    length_penalty = 1.0
    min_new_tokens = 0


class SCSTConfig:
    """SCST — giai đoạn 2 sau XE (train_scst.py), policy gradient với reward
    CIDEr-D (df toàn tập train). Biến thể Luo 2020: sample K caption/video,
    baseline = mean reward của K mẫu CÙNG video (không cần lượt greedy).
    Đơn vị lặp là VIDEO (1200/epoch) chứ không phải cặp video-caption."""
    # True: `python train.py` chạy trọn pipeline XE -> SCST trong MỘT lệnh
    # (SCST bắt đầu từ best-val ckpt của XE). False: chỉ XE như cũ; SCST vẫn
    # chạy riêng được qua `python train_scst.py --ckpt <path>`.
    enabled = False
    epochs = 30
    batch_size = 16      # số VIDEO mỗi bước (mỗi video sinh num_samples caption)
    num_samples = 5      # K
    lr = 5e-6            # RL rất dễ phá model pretrained -> lr nhỏ hơn XE ~20x
    gradient_clip = 5.0


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
    scst        = SCSTConfig

    """ Optimization """
    epochs = 16
    batch_size = 64
    gradient_clip = 5.0 # None if not used
    lr = 1e-4

    # --- Optimizer ---
    # "adam"  : baseline — Adam + L2 coupled, wd=5e-6 (thực chất ~0, không regularize)
    # "adamw" : decoupled weight decay THẬT, nhắm vào overfit đã quan sát được
    #           (train loss giảm đều nhưng val CIDEr đỉnh sớm ở epoch 9-17).
    #           Chỉ decay tham số >= 2D (weight matrix); bias/norm/scalar không decay.
    optimizer_type = "adamw"
    assert optimizer_type in ["adam", "adamw"]
    weight_decay = 5e-6        # dùng cho "adam" (giữ nguyên baseline)
    adamw_weight_decay = 0.01  # dùng cho "adamw"; ablate 0.05 nếu có tín hiệu

    # --- LR scheduler (sau warmup) ---
    # "plateau": baseline — ReduceLROnPlateau theo val loss. LƯU Ý: chưa từng
    #            kích hoạt trong mọi run 20 epoch (val loss giảm gần đơn điệu
    #            4.91->3.85, không plateau đủ patience) => thực tế là lr HẰNG SỐ.
    # "cosine" : cosine annealing lr -> lr*0.01 qua các epoch — bước nhỏ dần ở
    #            giai đoạn 9-17 nơi best model thường xuất hiện.
    scheduler_type = "cosine"
    assert scheduler_type in ["plateau", "cosine"]
    lr_decay_gamma = 0.5       # chỉ dùng cho "plateau"
    lr_decay_patience = 3      # chỉ dùng cho "plateau"
    # Fine-tune pretrained Flan-T5 decoder nên không cần warmup (0 = tắt;
    # train from scratch mới cần)
    warmup_epochs = 0
    label_smoothing = 0.15
    beam_size = 4
    # Số video gom vào một lần generate() lúc eval. Chỉ ảnh hưởng TỐC ĐỘ, không
    # đổi kết quả (mỗi video có mask GOP riêng). Bộ nhớ ~ eval_batch_size x
    # beam_size chuỗi cùng lúc -> giảm nếu OOM với decoder lớn (base/large/xl).
    eval_batch_size = 32

    """ Evaluation Metrics """
    metrics = ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']

    """ ID """
    feat_id = f"FEAT {feat.feature_spec} "\
              f"gop-{loader.num_gop}" + (f" {feat.raw_id}" if feat.raw_id else "")

    transformer_id = f"T5 "\
                     f"{transformer.t5_model_name} " \
                     f"mct-{transformer.max_caption_tokens} " \
                     f"dp-{transformer.dropout} " \
                     f"dl-{transformer.num_decoder_layers} " \
                     f"nrng-{transformer.no_repeat_ngram_size} " \
                     f"lp-{transformer.length_penalty} " \
                     f"mnt-{transformer.min_new_tokens}"

    optimizer_id = f"OPTIM {optimizer_type}+{scheduler_type} " \
                   f"lr-{lr} warmup-{warmup_epochs} " \
                   f"gamma-{lr_decay_gamma} pat-{lr_decay_patience} " \
                   f"wd-{weight_decay if optimizer_type == 'adam' else adamw_weight_decay}"

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
    print(f"Feature names     : {C.feat.feature_names}")
    print(f"Feature dimensions: {C.feat.feature_dims}")
    print(f"Raw features      : {list(C.feat.raw_feature_cfgs)}")
    print(f"Num GOP per video : {C.loader.num_gop}")
