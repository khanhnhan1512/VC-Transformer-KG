# coding=utf-8
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from models.t5_captioner import PositionalEncoding


class CLIPT5Captioner(nn.Module):
    """End-to-end video captioner: keyframes -> CLIP ViT -> Flan-T5 decoder.

    The T5 encoder is bypassed: projected CLIP frame embeddings are fed
    directly to the T5 decoder as `encoder_outputs`, same as T5Captioner.
    """

    SUPPORTED_TOKEN_MODES = ['cls']

    def __init__(self, clip_model_name, t5_model_name, dropout,
                 token_mode='cls',
                 freeze_vision_encoder=False,
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

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        t5_d_model = self.t5.config.d_model

        self.proj = nn.Sequential(
            nn.LayerNorm(clip_hidden),
            nn.Dropout(dropout),
            nn.Linear(clip_hidden, t5_d_model),
        )
        self.pos_embed = PositionalEncoding(t5_d_model, dropout, max_len=256)
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
