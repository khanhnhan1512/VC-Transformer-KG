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


class TypeEmbedding(nn.Module):
    """Feature type embedding: phân biệt token của các modality khác nhau
    (các pre-extracted feature, motion token, ...)."""

    def __init__(self, num_types, d_model):
        super().__init__()
        self.type_embeddings = nn.Embedding(num_types, d_model)
        nn.init.trunc_normal_(self.type_embeddings.weight, std=0.02)

    def forward(self, x, type_id):
        type_ids = torch.full(x.shape[:2], type_id, dtype=torch.long, device=x.device)
        return x + self.type_embeddings(type_ids)


class PositionalEncoding(nn.Module):
    """Sinusoidal PE theo CHỈ SỐ thứ tự GOP."""

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

    def forward(self, x, timestamps=None):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class TimestampEncoding(nn.Module):
    """Sinusoidal PE theo THỜI GIAN THẬT (giây) của I-frame mở đầu GOP.

    GOP được lấy mẫu không đều (I-frame định kỳ + scene-cut) nên khoảng cách
    thời gian thật mang thông tin mà chỉ số thứ tự không có.
    """

    def __init__(self, d_model, dropout):
        super().__init__()
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, timestamps):
        # timestamps: (B, T) giây -> pe: (B, T, d_model)
        angles = timestamps.unsqueeze(-1) * self.div_term  # (B, T, d_model/2)
        pe = torch.zeros(*timestamps.shape, self.d_model, device=x.device, dtype=x.dtype)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)
        return self.dropout(x + pe)


class MotionEncoder(nn.Module):
    """Mã hóa MV map của 1 GOP (2, g, g) thành 1 motion token (d_model).

    MV map là lưới displacement trung bình của các P/B-frame trong GOP —
    tín hiệu chuyển động lấy thẳng từ compressed domain, không cần optical flow.
    """

    def __init__(self, d_model, dropout, grid_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model),
        )

    def forward(self, motion_maps):
        # (B, T, 2, g, g) -> (B, T, d_model)
        B, T = motion_maps.size(0), motion_maps.size(1)
        flat = motion_maps.reshape(B * T, *motion_maps.shape[2:])
        return self.net(flat).view(B, T, -1)


class T5Captioner(nn.Module):
    """GOP-structured video captioner: pre-extracted features + motion -> Flan-T5 decoder.

    Mỗi GOP đóng góp: 1 token / pre-extracted feature (BLIP-2 QFormer, chỉ qua
    projection) + 1 motion token (MotionEncoder built-in trên MV map, train từ
    đầu). Các token xen kẽ [a₁, m₁, a₂, m₂, ...] với TypeEmbedding phân biệt
    modality, rồi đưa thẳng vào T5 decoder qua `encoder_outputs`. Khối T5
    encoder không được dùng nên bị loại bỏ ngay khi khởi tạo.

    Mask suy từ feature anchor (modality 0 — BLIP-2): vector thật của QFormer
    không bao giờ toàn 0, padding thì đúng bằng 0; vị trí pad của mọi modality
    trùng nhau nhờ chuẩn hóa độ dài chung trong dataset -> copy mask anchor
    cho mọi modality rồi interleave.
    """

    SUPPORTED_POS_ENCODINGS = ['index', 'timestamp']

    def __init__(self, d_feat, t5_model_name, dropout,
                 num_decoder_layers=0,
                 use_motion_tokens=False,
                 motion_grid_size=16,
                 pos_encoding_type='index',
                 device='cuda'):
        super().__init__()
        if pos_encoding_type not in self.SUPPORTED_POS_ENCODINGS:
            raise ValueError(
                f"[T5Captioner] Unsupported pos_encoding_type: '{pos_encoding_type}'. "
                f"Supported: {self.SUPPORTED_POS_ENCODINGS}")
        self.device = device
        self.use_motion_tokens = use_motion_tokens
        self.num_hdf5_feats = len(d_feat)

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

        self.feat_embeds = nn.ModuleList([
            FeatEmbedding(d_f, t5_d_model, dropout) for d_f in d_feat
        ])

        self.motion_encoder = None
        num_types = len(d_feat)
        if use_motion_tokens:
            self.motion_encoder = MotionEncoder(t5_d_model, dropout, grid_size=motion_grid_size)
            num_types += 1

        self.type_embed = TypeEmbedding(num_types, t5_d_model)
        if pos_encoding_type == 'timestamp':
            self.pos_embed = TimestampEncoding(t5_d_model, dropout)
        else:
            self.pos_embed = PositionalEncoding(t5_d_model, dropout, max_len=256)
        self.modality_norms = nn.ModuleList([
            nn.LayerNorm(t5_d_model) for _ in range(num_types)
        ])

    def _split_src(self, src):
        """src = (*hdf5_feats, motion_maps, timestamps) từ GOP loader,
        hoặc (*hdf5_feats,) từ loader cũ (không motion/timestamp)."""
        n = self.num_hdf5_feats
        hdf5_feats = src[:n]
        motion_maps = src[n] if len(src) > n else None
        timestamps = src[n + 1] if len(src) > n + 1 else None
        return hdf5_feats, motion_maps, timestamps

    def encode(self, src):
        hdf5_feats, motion_maps, timestamps = self._split_src(src)

        tokens = []
        for i, feat in enumerate(hdf5_feats):
            x = self.feat_embeds[i](feat)
            x = self.type_embed(x, type_id=i)
            x = self.pos_embed(x, timestamps)
            x = self.modality_norms[i](x)
            tokens.append(x)

        if self.motion_encoder is not None:
            m = self.motion_encoder(motion_maps)
            m = self.type_embed(m, type_id=self.num_hdf5_feats)
            m = self.pos_embed(m, timestamps)  # cùng vị trí GOP với token feature
            m = self.modality_norms[self.num_hdf5_feats](m)
            tokens.append(m)

        # Interleave: [a¹₁, ..., m₁, a¹₂, ..., m₂, ...] -> (B, num_types*T, D)
        B, _, D = tokens[0].shape
        stacked = torch.stack(tokens, dim=2)
        encoder_hidden = stacked.reshape(B, -1, D)

        # Mask từ anchor (modality 0), copy cho mọi modality rồi interleave
        anchor = hdf5_feats[0]
        gop_mask = (anchor.abs().sum(dim=-1) > 0)               # (B, T)
        attention_mask = gop_mask.unsqueeze(-1) \
            .expand(-1, -1, len(tokens)).reshape(B, -1).long()  # (B, num_types*T)

        return encoder_hidden, attention_mask

    def forward(self, src, labels=None, decoder_attention_mask=None):
        encoder_hidden, attention_mask = self.encode(src)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return outputs

    def generate_captions(self, src, tokenizer, beam_size, max_len):
        encoder_hidden, attention_mask = self.encode(src)

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
