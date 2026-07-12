# coding=utf-8
import math
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def _get_vision_transformer(vision_encoder):
    """Lấy module CLIPVisionTransformer bên trong CLIPVisionModel.

    Tương thích nhiều phiên bản transformers: v4.x đặt tên `.vision_model`,
    v5.x đổi thành `.model`; fallback cuối dùng `.base_model`.
    """
    for attr in ('vision_model', 'model'):
        inner = getattr(vision_encoder, attr, None)
        if inner is not None and hasattr(inner, 'encoder'):
            return inner
    return vision_encoder.base_model


def select_evenly_spaced_layers(layers, num_keep):
    """Chọn num_keep layer cách đều (linspace) từ ModuleList pretrained.

    Luôn bao gồm layer đầu tiên (index 0) — quan trọng với T5 decoder vì
    relative_attention_bias chỉ nằm ở block 0 và được các block sau dùng chung.
    """
    idxs = torch.linspace(0, len(layers) - 1, steps=num_keep).round().long().tolist()
    return nn.ModuleList([layers[i] for i in idxs]), idxs


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embedding theo CHỈ SỐ thứ tự GOP.

    Số vị trí nhỏ và cố định (keyframe_threshold) nên learnable phù hợp hơn
    sinusoidal — không cần khả năng ngoại suy chuỗi dài.
    """

    def __init__(self, d_model, dropout, max_len=64):
        super().__init__()
        self.pos_embeddings = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pos_embeddings.weight, std=0.02)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, timestamps=None):
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_embeddings(positions)
        return self.dropout(x)


class TimestampEncoding(nn.Module):
    """Sinusoidal positional encoding theo THỜI GIAN THẬT (giây) của I-frame.

    Keyframe được lấy mẫu không đều (I-frame định kỳ + scene-cut) nên khoảng
    cách thời gian thật mang thông tin mà chỉ số thứ tự không có.
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


class TypeEmbedding(nn.Module):
    """Feature type embedding: phân biệt token appearance (0) và motion (1)."""

    def __init__(self, num_types, d_model):
        super().__init__()
        self.type_embeddings = nn.Embedding(num_types, d_model)
        nn.init.trunc_normal_(self.type_embeddings.weight, std=0.02)

    def forward(self, x, type_id):
        type_ids = torch.full(x.shape[:2], type_id, dtype=torch.long, device=x.device)
        return x + self.type_embeddings(type_ids)


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
            nn.LayerNorm(d_model),
        )

    def forward(self, motion_maps):
        # (B, T, 2, g, g) -> (B, T, d_model)
        B, T = motion_maps.size(0), motion_maps.size(1)
        flat = motion_maps.reshape(B * T, *motion_maps.shape[2:])
        return self.net(flat).view(B, T, -1)


class CLIPT5Captioner(nn.Module):
    """End-to-end video captioner theo cấu trúc GOP.

    Mỗi GOP -> token appearance (CLS của ViT trên I-frame) + token motion
    (MotionEncoder trên MV map của các P/B-frame). TypeEmbedding phân biệt
    hai loại token; positional encoding theo thứ tự GOP hoặc timestamp thật.
    The T5 encoder is bypassed: tokens are fed directly to the T5 decoder
    as `encoder_outputs`, same as T5Captioner.
    """

    SUPPORTED_TOKEN_MODES = ['cls']
    SUPPORTED_POS_ENCODINGS = ['index', 'timestamp']

    def __init__(self, clip_model_name, t5_model_name, dropout,
                 token_mode='cls',
                 freeze_vision_encoder=False,
                 num_vision_layers=0,
                 num_decoder_layers=0,
                 use_motion_tokens=True,
                 motion_grid_size=16,
                 pos_encoding_type='index',
                 device='cuda'):
        super().__init__()
        if token_mode not in self.SUPPORTED_TOKEN_MODES:
            raise ValueError(
                f"[CLIPT5Captioner] Unsupported token_mode: '{token_mode}'. "
                f"Supported: {self.SUPPORTED_TOKEN_MODES}")
        if pos_encoding_type not in self.SUPPORTED_POS_ENCODINGS:
            raise ValueError(
                f"[CLIPT5Captioner] Unsupported pos_encoding_type: '{pos_encoding_type}'. "
                f"Supported: {self.SUPPORTED_POS_ENCODINGS}")
        self.device = device
        self.token_mode = token_mode
        self.use_motion_tokens = use_motion_tokens
        self.pos_encoding_type = pos_encoding_type

        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        clip_hidden = self.vision_encoder.config.hidden_size

        # Chọn N layer cách đều của vision encoder (CLS vẫn đi qua post_layernorm của CLIP)
        vision_transformer = _get_vision_transformer(self.vision_encoder)
        vision_layers = vision_transformer.encoder.layers
        if 0 < num_vision_layers < len(vision_layers):
            selected, idxs = select_evenly_spaced_layers(vision_layers, num_vision_layers)
            vision_transformer.encoder.layers = selected
            self.vision_encoder.config.num_hidden_layers = num_vision_layers
            print(f"[CLIPT5Captioner] Vision encoder giữ các layer: {idxs}")

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        t5_d_model = self.t5.config.d_model

        if 0 < num_decoder_layers < len(self.t5.decoder.block):
            selected, idxs = select_evenly_spaced_layers(self.t5.decoder.block, num_decoder_layers)
            self.t5.decoder.block = selected
            self.t5.config.num_decoder_layers = num_decoder_layers
            print(f"[CLIPT5Captioner] T5 decoder giữ các block: {idxs}")

        # T5 encoder không bao giờ được dùng (forward/generate luôn truyền encoder_outputs)
        # -> bỏ toàn bộ block để model gọn (~19M params); embed_tokens là shared với decoder nên giữ nguyên
        self.t5.encoder.block = nn.ModuleList()
        self.t5.config.num_layers = 0

        # pooler_output đã qua post_layernorm của CLIP nên không cần LayerNorm ở đây
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(clip_hidden, t5_d_model),
        )

        self.motion_encoder = None
        if use_motion_tokens:
            self.motion_encoder = MotionEncoder(t5_d_model, dropout, grid_size=motion_grid_size)

        self.type_embed = TypeEmbedding(num_types=2, d_model=t5_d_model)
        if pos_encoding_type == 'timestamp':
            self.pos_embed = TimestampEncoding(t5_d_model, dropout)
        else:
            self.pos_embed = LearnablePositionalEncoding(t5_d_model, dropout, max_len=32)
        self.final_norm = nn.LayerNorm(t5_d_model)

        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def encode(self, pixel_values, motion_maps, timestamps, frame_mask):
        """GOP data -> (encoder_hidden (B, L, d_model), attention_mask (B, L)).

        L = T (chỉ appearance) hoặc 2T (xen kẽ a1,m1,a2,m2,... khi có motion).
        All frames (real and padded) go through the vision encoder for
        simplicity; padded positions are blocked from the decoder by the
        cross-attention mask.
        """
        B, T = pixel_values.size(0), pixel_values.size(1)
        flat = pixel_values.reshape(B * T, *pixel_values.shape[2:])

        vision_outputs = self.vision_encoder(pixel_values=flat)
        # pooler_output = post-layernorm CLS token, one per frame
        cls_tokens = vision_outputs.pooler_output.view(B, T, -1)

        a = self.proj(cls_tokens)                     # token appearance (B, T, D)
        a = self.type_embed(a, type_id=0)
        a = self.pos_embed(a, timestamps)

        if self.motion_encoder is not None:
            m = self.motion_encoder(motion_maps)      # token motion (B, T, D)
            m = self.type_embed(m, type_id=1)
            # Cùng vị trí GOP với token appearance tương ứng
            m = self.pos_embed(m, timestamps)

            # Xen kẽ [a1, m1, a2, m2, ...] -> (B, 2T, D); mask expand tương ứng
            x = torch.stack([a, m], dim=2).reshape(B, 2 * T, -1)
            attention_mask = frame_mask.unsqueeze(-1).expand(B, T, 2).reshape(B, 2 * T)
        else:
            x = a
            attention_mask = frame_mask

        x = self.final_norm(x)
        return x, attention_mask

    def forward(self, src, labels=None, decoder_attention_mask=None):
        pixel_values, motion_maps, timestamps, frame_mask = src
        encoder_hidden, attention_mask = self.encode(
            pixel_values, motion_maps, timestamps, frame_mask)

        # T5 tràn số (NaN) khi chạy fp16 autocast (pretrain bằng bf16, activation
        # trong T5DenseGatedActDense vượt ngưỡng fp16) -> luôn chạy T5 ở fp32.
        # AMP vẫn áp dụng cho vision encoder (phần chiếm phần lớn compute).
        with torch.autocast('cuda', enabled=False):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden.float())
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
        return outputs

    def generate_captions(self, src, tokenizer, beam_size, max_len):
        pixel_values, motion_maps, timestamps, frame_mask = src
        encoder_hidden, attention_mask = self.encode(
            pixel_values, motion_maps, timestamps, frame_mask)

        # Giống forward: T5 luôn chạy fp32 để tránh tràn số fp16
        with torch.autocast('cuda', enabled=False):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden.float())
            generated_ids = self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                num_beams=beam_size,
                max_length=max_len,
                early_stopping=True,
            )
        captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return captions
