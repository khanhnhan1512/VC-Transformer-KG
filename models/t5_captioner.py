import math
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def select_evenly_spaced_layers(layers, num_keep):
    """Chọn num_keep layer cách đều (linspace) từ ModuleList pretrained.

    Luôn bao gồm layer đầu tiên (index 0) — quan trọng với T5 vì
    relative_attention_bias chỉ nằm ở block 0 và được các block sau dùng chung.
    (Layer cuối cũng luôn được giữ do linspace kết thúc tại len-1.)
    """
    idxs = torch.linspace(0, len(layers) - 1, steps=num_keep).round().long().tolist()
    return nn.ModuleList([layers[i] for i in idxs]), idxs


class FeatEmbedding(nn.Module):
    """Projection: đưa feature về đúng kích thước d_model của T5 decoder."""

    def __init__(self, d_feat, d_model, dropout):
        super().__init__()
        self.embeddings = nn.Sequential(
            nn.LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_model)
        )

    def forward(self, x):
        return self.embeddings(x)


class FeatureTypeEmbedding(nn.Module):
    """Phân biệt token thuộc loại feature nào.

    Với GOP-structured input, hai token của cùng một GOP có cùng positional
    encoding (cùng chỉ số GOP) — chính embedding này là thứ duy nhất phân biệt
    token appearance (I-frame) với token motion (MV) của GOP đó.
    """

    def __init__(self, num_types, d_model):
        super().__init__()
        self.type_embeddings = nn.Embedding(num_types, d_model)

    def forward(self, x, type_ids):
        return x + self.type_embeddings(type_ids)


def pool_motion_bins(motion, pool_bins):
    """Gộp K bin thời gian của mỗi GOP xuống còn `pool_bins` bin.

    Motion được lưu ở K=8 để ablate K mà không phải trích lại. Nhưng mỗi bin
    lưu TRUNG BÌNH THEO Ô, nên gộp 2 trung bình != trung bình của hợp — phải
    lấy trung bình CÓ TRỌNG SỐ theo số MV. Kênh density (index 3) tỉ lệ thuận
    với số MV trong ô (vì các bin có số frame xấp xỉ bằng nhau), nên dùng nó
    làm trọng số.

    motion: (B, T, K, C, G, G) -> (B, T, pool_bins, C, G, G)
    """
    B, T, K, C, G, _ = motion.shape
    assert K % pool_bins == 0, f"K={K} không chia hết cho pool_bins={pool_bins}"
    if pool_bins == K:
        return motion

    g = K // pool_bins
    m = motion.reshape(B, T, pool_bins, g, C, G, G)
    w = m[:, :, :, :, 3:4]                                    # density (B,T,P,g,1,G,G)
    vals = (m[:, :, :, :, :3] * w).sum(dim=3) / w.sum(dim=3).clamp(min=1e-6)
    dens = m[:, :, :, :, 3:4].mean(dim=3)                     # density gộp = trung bình thường
    return torch.cat([vals, dens], dim=3)                     # (B,T,P,C,G,G)


class MotionEncoder(nn.Module):
    """Motion vector grid THÔ của mỗi GOP -> 1 token d_out. (B,T,K,C,G,G) -> (B,T,d_out)

    Input: K bin thời gian trong GOP, C=4 kênh (dx, dy, |v|, density), lưới GxG.
    Conv3D quét đồng thời trục thời gian (K) và không gian (G,G) để học động lực
    chuyển động bên trong GOP.

    Dùng GroupNorm chứ không BatchNorm: batch chứa các GOP pad toàn 0 (video ít
    GOP hơn num_gop), thống kê batch sẽ bị chúng làm nhiễu.

    `proj` là Linear tối giản (pool -> flatten -> Linear). Nó KHÔNG thừa: nếu bỏ
    đi thì conv cuối phải xuất thẳng d_out kênh (3x3x3 x 64 x 512 = 885K params)
    thay vì 128 kênh + Linear 128->512 (65K) — tức bỏ Linear lại làm encoder
    PHÌNH ~3.5 lần. Không có Dropout ở đây vì FeatEmbedding ngay sau đã có.

    GROUNDING (context_dim != None): motion một mình bị "mù" — trường dịch
    chuyển "khối 30x40px trôi sang phải" không phân biệt được con chó chạy hay
    camera lia. FiLM (Perez et al. 2018) điều biến từng kênh conv bằng
    (1+gamma)*x + beta, với gamma/beta sinh từ APPEARANCE TOKEN CÙNG GOP —
    encoder được biết "cái gì đang chuyển động" ngay từ lúc encode, thay vì đợi
    đến cross-attention của decoder. (Ngữ cảnh chỉ có 1 token/GOP nên
    cross-attention thoái hóa thành cộng có trọng số — FiLM là công cụ đúng.)
    FiLM head khởi tạo 0 -> gamma=beta=0 -> lúc bắt đầu train hành vi Y HỆT
    bản không grounding, sau đó model tự học mức điều biến cần thiết.
    """

    def __init__(self, d_out, in_channels=4, num_bins=8, grid_size=16,
                 pool_bins=None, width=32, context_dim=None, film_hidden=128):
        super().__init__()
        self.pool_bins = pool_bins or num_bins
        self.num_bins = num_bins
        self.d_out = d_out

        w1, w2, w3 = width, width * 2, width * 4
        # Tách 3 stage (Conv->GroupNorm) để chèn FiLM sau norm, trước ReLU
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv3d(in_channels, w1, kernel_size=3, padding=1),
                          nn.GroupNorm(4, w1)),
            nn.Sequential(nn.Conv3d(w1, w2, kernel_size=3, stride=2, padding=1),
                          nn.GroupNorm(8, w2)),
            nn.Sequential(nn.Conv3d(w2, w3, kernel_size=3, stride=2, padding=1),
                          nn.GroupNorm(8, w3)),
        ])
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.proj = nn.Linear(w3, d_out)

        self.film_trunk = None
        if context_dim is not None:
            # Bottleneck chung (context_dim -> film_hidden) + head riêng mỗi stage
            # (-> gamma||beta), để chi phí không phình theo context_dim lớn (1536)
            self.film_trunk = nn.Sequential(
                nn.Linear(context_dim, film_hidden), nn.ReLU(inplace=True))
            self.film_heads = nn.ModuleList(
                [nn.Linear(film_hidden, 2 * w) for w in (w1, w2, w3)])
            for h in self.film_heads:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

    def forward(self, motion, context=None):
        """motion (B,T,K,C,G,G) [+ context (B,T,D_app)] -> (B,T,d_out)"""
        motion = pool_motion_bins(motion, self.pool_bins)
        B, T = motion.shape[:2]
        x = motion.reshape(B * T, *motion.shape[2:])   # (B*T, K, C, G, G)
        x = x.permute(0, 2, 1, 3, 4).contiguous()      # (B*T, C, K, G, G) cho Conv3d

        film_params = None
        if self.film_trunk is not None:
            assert context is not None, "MotionEncoder grounded nhưng không nhận context"
            h = self.film_trunk(context.reshape(B * T, -1))
            film_params = [head(h) for head in self.film_heads]

        for s, stage in enumerate(self.stages):
            x = stage(x)                               # Conv -> GroupNorm
            if film_params is not None:
                gamma, beta = film_params[s].chunk(2, dim=1)
                x = x * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
            x = torch.relu(x)

        return self.proj(self.head(x)).reshape(B, T, self.d_out)


# Feature THÔ (chưa qua encoder nào lúc trích) -> cần encoder học được trong model.
# Tên khớp với modality name trong FeatureConfig.model.
RAW_ENCODERS = {
    "MotionMV": MotionEncoder,
}


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class T5Captioner(nn.Module):
    """Pre-extracted video features -> projection -> Flan-T5 decoder.

    Không dùng encoder/fusion layer nào để biến đổi feature: mỗi feature chỉ
    qua projection (+ segment/positional embedding) rồi đưa thẳng vào T5
    decoder qua `encoder_outputs` (ý tưởng BiDecT, thay bidirectional decoder
    bằng pretrained Flan-T5 decoder). Khối T5 encoder không được dùng nên bị
    loại bỏ ngay khi khởi tạo.
    """

    def __init__(self, d_feat, t5_model_name, dropout,
                 num_decoder_layers=0,
                 feature_names=None,
                 raw_feature_cfgs=None,
                 device='cuda'):
        super().__init__()
        self.device = device

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        t5_d_model = self.t5.config.d_model

        # Giữ N block decoder cách đều (linspace, luôn gồm block 0 mang relative bias)
        if 0 < num_decoder_layers < len(self.t5.decoder.block):
            selected, idxs = select_evenly_spaced_layers(
                self.t5.decoder.block, num_decoder_layers)
            self.t5.decoder.block = selected
            self.t5.config.num_decoder_layers = num_decoder_layers
            print(f"[T5Captioner] T5 decoder giữ các block: {idxs}")

        # T5 encoder không bao giờ được dùng (forward/generate luôn truyền encoder_outputs)
        # -> bỏ toàn bộ block để model gọn; embed_tokens là shared với decoder nên giữ nguyên
        self.t5.encoder.block = nn.ModuleList()
        self.t5.config.num_layers = 0

        # Mỗi modality có 1 encoder đứng trước projection:
        #   - feature pre-extracted (đã sẵn sàng project) -> Identity
        #   - feature THÔ (motion vector grid) -> encoder học được, đưa về d_feat[i]
        # Sau bước này MỌI modality đều là (B, T, d_i) -> vòng lặp trong encode()
        # xử lý đồng nhất, không cần biết cái nào thô cái nào không.
        feature_names = feature_names or [f"feat{i}" for i in range(len(d_feat))]
        raw_feature_cfgs = raw_feature_cfgs or {}
        assert len(feature_names) == len(d_feat), \
            f"feature_names ({len(feature_names)}) phải cùng độ dài d_feat ({len(d_feat)})"

        self.is_raw = [name in raw_feature_cfgs for name in feature_names]
        # Feature pre-extracted đầu tiên đóng 2 vai: nguồn GOP mask, và ngữ cảnh
        # (appearance token) cho các raw encoder grounded
        self.context_ref = next(
            (i for i, r in enumerate(self.is_raw) if not r), None)

        self.encoders = nn.ModuleList()
        for i, name in enumerate(feature_names):
            if not self.is_raw[i]:
                self.encoders.append(nn.Identity())
                continue
            cfg = dict(raw_feature_cfgs[name])
            assert cfg["d_out"] == d_feat[i], (
                f"{name}: d_out={cfg['d_out']} không khớp feature_dims[{i}]={d_feat[i]}")
            grounded = cfg.pop("grounded", False)
            if grounded:
                assert self.context_ref is not None, \
                    f"'{name}' grounded cần ít nhất 1 feature pre-extracted làm ngữ cảnh"
                # context_dim suy tự động từ appearance feature -> đổi appearance
                # trong feature_spec không phải sửa gì thêm
                cfg["context_dim"] = d_feat[self.context_ref]
            self.encoders.append(RAW_ENCODERS[name](**cfg))
            print(f"[T5Captioner] '{name}' là feature THÔ -> "
                  f"{RAW_ENCODERS[name].__name__}(d_out={cfg['d_out']}, "
                  f"grounded={'trên ' + feature_names[self.context_ref] if grounded else False})")

        self.feat_embeds = nn.ModuleList([
            FeatEmbedding(d_f, t5_d_model, dropout) for d_f in d_feat
        ])
        self.type_embed = FeatureTypeEmbedding(len(d_feat), t5_d_model)
        self.pos_embed = PositionalEncoding(t5_d_model, dropout, max_len=256)
        self.feat_norms = nn.ModuleList([
            nn.LayerNorm(t5_d_model) for _ in d_feat
        ])

    def encode(self, src):
        batch_size = src[0].size(0)
        feats = []
        for i, feat in enumerate(src):
            if self.is_raw[i]:
                # Feature THÔ -> (B,T,d_i); encoder nhận kèm appearance token cùng
                # GOP làm ngữ cảnh (tự bỏ qua nếu không grounded)
                ctx = src[self.context_ref] if self.context_ref is not None else None
                feat = self.encoders[i](feat, ctx)
            type_id = torch.full(
                (batch_size, feat.size(1)), i, dtype=torch.long, device=self.device
            )
            x = self.feat_embeds[i](feat)
            x = self.type_embed(x, type_id)
            x = self.pos_embed(x)
            x = self.feat_norms[i](x)
            feats.append(x)

        # stack(dim=2) + reshape -> interleave theo GOP: [a1, m1, a2, m2, ...]
        B, _, D = feats[0].shape
        stacked = torch.stack(feats, dim=2)
        return stacked.reshape(B, -1, D)

    def _build_encoder_attention_mask(self, src):
        """Mask ở mức GOP: GOP nào là thật, GOP nào là zero-pad.

        Mask là thuộc tính của GOP, KHÔNG phải của modality: loader đã assert mọi
        feature của cùng 1 video có cùng NUM_GOP và được pad/sample cùng chỉ số.
        Nên chỉ dựng MỘT mask rồi nhân bản cho mọi modality — hai token của cùng
        một GOP vì thế luôn cùng số phận (không thể có chuyện appearance bị mask
        còn motion thì không).

        Dựng từ feature pre-extracted đầu tiên. KHÔNG bao giờ suy từ feature THÔ:
          - 119/11303 GOP THẬT có 0 P/B-frame -> motion toàn 0 nhưng GOP vẫn hợp
            lệ (có I-frame) -> sẽ bị loại nhầm;
          - output encoder với input toàn 0 cũng không phải 0 (conv có bias).
        """
        assert self.context_ref is not None, \
            "Cần ít nhất 1 feature pre-extracted để dựng GOP mask (không thể suy từ feature THÔ)"

        gop_mask = (src[self.context_ref].abs().sum(dim=-1) > 0)  # (B, num_gop)

        # Mỗi GOP sinh ra len(src) token liên tiếp sau interleave -> nhân bản mask
        # theo đúng thứ tự stack(dim=2).reshape() ở encode()
        B, T = gop_mask.shape
        return gop_mask.unsqueeze(2).expand(B, T, len(src)).reshape(B, -1).long()

    def forward(self, src, labels=None, decoder_attention_mask=None):
        encoder_hidden = self.encode(src)
        attention_mask = self._build_encoder_attention_mask(src)

        # Tự dựng decoder_input_ids thay vì truyền `labels=` xuống T5: khi nhận
        # labels, T5 tính CrossEntropyLoss BÊN TRONG — mà ta luôn vứt đi để tính
        # lại bằng loss_fct riêng (cần label_smoothing). Bỏ `labels=` cho logits
        # Y HỆT nhưng tránh một lượt log_softmax thừa trên (B, L, 32k) cùng ~100MB
        # activation bị giữ lại vô ích. `outputs.loss` giờ là None — không ai dùng.
        decoder_input_ids = (self.t5.prepare_decoder_input_ids_from_labels(labels)
                             if labels is not None else None)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        return outputs

    def generate_captions(self, src, tokenizer, beam_size, max_len):
        encoder_hidden = self.encode(src)
        attention_mask = self._build_encoder_attention_mask(src)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        generated_ids = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            num_beams=beam_size,
            max_length=max_len,
            early_stopping=True,
        )
        captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return captions

    # ===== SCST (self-critical sequence training, giai đoạn 2 sau XE) =====

    def encode_with_mask(self, src):
        """Encoder pass gọi MỘT lần cho cả sampling lẫn scoring.

        generate() chạy trong no_grad nội bộ nên không giữ graph; tensor trả về
        ở đây vẫn mang grad_fn -> scoring pass (sequence_logprobs) tái dùng đúng
        graph này để gradient chảy về cả projection/type-embed/motion encoder.
        """
        return self.encode(src), self._build_encoder_attention_mask(src)

    def sample_captions(self, encoder_hidden, attention_mask, num_samples, max_len):
        """Multinomial sampling K caption/video (top_k=0 = sampling thuần, đúng
        chuẩn SCST). Trả (B*K, L); K dòng của cùng video nằm LIÊN TIẾP
        (repeat_interleave của HF generate)."""
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        return self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            do_sample=True, top_k=0, top_p=1.0, temperature=1.0,
            num_return_sequences=num_samples,
            max_length=max_len,
        )

    def sequence_logprobs(self, encoder_hidden, attention_mask, sequences):
        """Teacher-forcing CÓ GRAD trên chuỗi đã sample -> (logp, mask) per token.

        sequences: (N, L) như generate trả về — mở đầu bằng decoder_start_token
        (T5 dùng pad làm start). decoder_input = seq[:, :-1], label = seq[:, 1:]
        -> logits[t] dự đoán đúng token t+1. Pad sau EOS bị mask (EOS id=1 != pad
        id=0 nên EOS vẫn được tính — quan trọng: model phải học cả lúc DỪNG).
        """
        pad_id = self.t5.config.pad_token_id
        labels = sequences[:, 1:]
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        logits = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=sequences[:, :-1],
        ).logits
        logp = logits.log_softmax(dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
        token_mask = (labels != pad_id).float()
        return logp, token_mask
