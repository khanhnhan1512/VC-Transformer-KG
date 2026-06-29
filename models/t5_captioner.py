import math
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        self.original_linear = original_linear
        self.scaling = alpha / r

        for param in self.original_linear.parameters():
            param.requires_grad = False

        in_f = original_linear.in_features
        out_f = original_linear.out_features
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original_linear(x) + self.lora_B(self.lora_A(x)) * self.scaling


class FeatEmbedding(nn.Module):
    def __init__(self, d_feat, d_model, dropout):
        super().__init__()
        self.embeddings = nn.Sequential(
            nn.LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_model)
        )

    def forward(self, x):
        return self.embeddings(x)


class SegmentEmbedding(nn.Module):
    def __init__(self, num_segments, d_model):
        super().__init__()
        self.segment_embeddings = nn.Embedding(num_segments, d_model)

    def forward(self, x, segment_ids):
        return x + self.segment_embeddings(segment_ids)


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

    def __init__(self, d_feat, t5_model_name, dropout,
                 lora_r=0, lora_alpha=16, lora_target_modules=None,
                 device='cuda'):
        super().__init__()
        self.device = device

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        t5_d_model = self.t5.config.d_model

        self.feat_embeds = nn.ModuleList([
            FeatEmbedding(d_f, t5_d_model, dropout) for d_f in d_feat
        ])
        self.seg_embed = SegmentEmbedding(len(d_feat), t5_d_model)
        self.pos_embed = PositionalEncoding(t5_d_model, dropout, max_len=256)
        self.feat_norms = nn.ModuleList([
            nn.LayerNorm(t5_d_model) for _ in d_feat
        ])

        if lora_r > 0:
            self._apply_lora(lora_r, lora_alpha, lora_target_modules or ['q', 'v'])

    def _apply_lora(self, r, alpha, target_modules):
        for param in self.t5.parameters():
            param.requires_grad = False

        for block in self.t5.decoder.block:
            self_attn = block.layer[0].SelfAttention
            for name in target_modules:
                setattr(self_attn, name, LoRALinear(getattr(self_attn, name), r, alpha))

            cross_attn = block.layer[1].EncDecAttention
            for name in target_modules:
                setattr(cross_attn, name, LoRALinear(getattr(cross_attn, name), r, alpha))

    def encode(self, src):
        batch_size = src[0].size(0)
        feats = []
        for i, feat in enumerate(src):
            seg_id = torch.full(
                (batch_size, feat.size(1)), i, dtype=torch.long, device=self.device
            )
            x = self.feat_embeds[i](feat)
            x = self.seg_embed(x, seg_id)
            x = self.pos_embed(x)
            x = self.feat_norms[i](x)
            feats.append(x)

        B, _, D = feats[0].shape
        stacked = torch.stack(feats, dim=2)
        return stacked.reshape(B, -1, D)

    def _build_encoder_attention_mask(self, src):
        masks = []
        for feat in src:
            mask = (feat.abs().sum(dim=-1) > 0)
            masks.append(mask)
        stacked = torch.stack(masks, dim=2)
        return stacked.reshape(masks[0].size(0), -1).long()

    def forward(self, src, labels=None, decoder_attention_mask=None):
        encoder_hidden = self.encode(src)
        attention_mask = self._build_encoder_attention_mask(src)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
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
