# coding=utf-8
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def select_evenly_spaced_layers(layers, num_keep):
    """Chọn num_keep layer cách đều (linspace) từ ModuleList pretrained.

    Luôn bao gồm layer đầu tiên (index 0) — quan trọng với T5 decoder vì
    relative_attention_bias chỉ nằm ở block 0 và được các block sau dùng chung.
    """
    idxs = torch.linspace(0, len(layers) - 1, steps=num_keep).round().long().tolist()
    return nn.ModuleList([layers[i] for i in idxs]), idxs


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embedding cho chuỗi keyframe (thứ tự thời gian).

    Số vị trí nhỏ và cố định (keyframe_threshold) nên learnable phù hợp hơn
    sinusoidal — không cần khả năng ngoại suy chuỗi dài.
    """

    def __init__(self, d_model, dropout, max_len=64):
        super().__init__()
        self.pos_embeddings = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pos_embeddings.weight, std=0.02)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_embeddings(positions)
        return self.dropout(x)


class CLIPT5Captioner(nn.Module):
    """End-to-end video captioner: keyframes -> CLIP ViT -> Flan-T5 decoder.

    The T5 encoder is bypassed: projected CLIP frame embeddings are fed
    directly to the T5 decoder as `encoder_outputs`, same as T5Captioner.
    """

    SUPPORTED_TOKEN_MODES = ['cls']

    def __init__(self, clip_model_name, t5_model_name, dropout,
                 token_mode='cls',
                 freeze_vision_encoder=False,
                 num_vision_layers=0,
                 num_decoder_layers=0,
                 device='cuda'):
        super().__init__()
        if token_mode not in self.SUPPORTED_TOKEN_MODES:
            raise ValueError(
                f"[CLIPT5Captioner] Unsupported token_mode: '{token_mode}'. "
                f"Supported: {self.SUPPORTED_TOKEN_MODES}")
        self.device = device
        self.token_mode = token_mode

        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        clip_hidden = self.vision_encoder.config.hidden_size

        # Chọn N layer cách đều của vision encoder (CLS vẫn đi qua post_layernorm của CLIP)
        vision_layers = self.vision_encoder.vision_model.encoder.layers
        if 0 < num_vision_layers < len(vision_layers):
            selected, idxs = select_evenly_spaced_layers(vision_layers, num_vision_layers)
            self.vision_encoder.vision_model.encoder.layers = selected
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
        self.pos_embed = LearnablePositionalEncoding(t5_d_model, dropout, max_len=32)
        self.final_norm = nn.LayerNorm(t5_d_model)

        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def encode(self, pixel_values):
        """(B, T, 3, H, W) -> (B, T, d_model).

        All frames (real and padded) go through the vision encoder for
        simplicity; padded positions are blocked from the decoder by the
        cross-attention mask built in the collate function.
        """
        B, T = pixel_values.size(0), pixel_values.size(1)
        flat = pixel_values.reshape(B * T, *pixel_values.shape[2:])

        vision_outputs = self.vision_encoder(pixel_values=flat)
        # pooler_output = post-layernorm CLS token, one per frame
        cls_tokens = vision_outputs.pooler_output.view(B, T, -1)

        x = self.proj(cls_tokens)
        x = self.pos_embed(x)
        x = self.final_norm(x)
        return x

    def forward(self, src, labels=None, decoder_attention_mask=None):
        pixel_values, frame_mask = src
        encoder_hidden = self.encode(pixel_values)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=frame_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return outputs

    def generate_captions(self, src, tokenizer, beam_size, max_len):
        pixel_values, frame_mask = src
        encoder_hidden = self.encode(pixel_values)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        generated_ids = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=frame_mask,
            num_beams=beam_size,
            max_length=max_len,
            early_stopping=True,
        )
        captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return captions
