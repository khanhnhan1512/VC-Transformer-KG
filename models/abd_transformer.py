import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable
from typing import List, Optional

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class FeatureFusion(nn.Module):
    """
    Feature fusion module for combining a list of feature tensors with identical shape
    (batch_size, seq_len, d_model) into a single tensor of shape (batch_size, seq_len, d_model).

    Args:
        d_model: model dimension (if None, inferred on first forward call for some methods).
        method: one of {'weighted', 'gated', 'concat_linear', 'self_attn'}.
        num_heads: used only for 'self_attn'.
        dropout: dropout applied to output.
        use_layernorm: apply LayerNorm after fusion.
        residual: add residual connection from first feature.
    """
    def __init__(self,
                 d_model: Optional[int] = None,
                 method: str = "gated",
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_layernorm: bool = True,
                 residual: bool = True):
        super().__init__()
        assert method in {"weighted", "gated", "concat_linear", "self_attn"}
        self.method = method
        self._d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_layernorm = use_layernorm
        self.residual = residual

        # lazy init components that depend on d_model / num_features
        # for 'weighted' we'll create weight logits when we know num_features
        # for 'gated' we create per-feature gating linear layers lazily
        self._initialized = False
        self._num_features = None

        self._weight_logits = None        # Parameter (n_features,) for 'weighted'
        self._gates = nn.ModuleList()     # list of Linear(d_model, d_model) for 'gated'
        self._concat_proj = None          # Linear(n_features * d_model -> d_model)
        self._attn = None                 # MultiheadAttention for 'self_attn'
        self.out_proj = None              # optional final projection (keeps d_model stable)
        self.ln = None

    def _lazy_init(self, sample_tensor: torch.Tensor, num_features: int):
        d_model = self._d_model or sample_tensor.size(-1)
        self._d_model = d_model
        self._num_features = num_features

        if self.method == "weighted":
            self._weight_logits = nn.Parameter(torch.zeros(num_features))  # learnable logits
        elif self.method == "gated":
            # per-feature gating linear (maps d_model -> d_model) so gate is elementwise sigmoid
            self._gates = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_features)])
        elif self.method == "concat_linear":
            self._concat_proj = nn.Linear(num_features * d_model, d_model)
        elif self.method == "self_attn":
            # We'll concatenate features along sequence dim and run multihead self-attention
            self._attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=self.num_heads, batch_first=True)
            # no extra params besides attn
        # final projection to stabilize representation (identity if not needed)
        self.out_proj = nn.Linear(d_model, d_model)
        if self.use_layernorm:
            self.ln = nn.LayerNorm(d_model)

        # Move newly-created parameters/modules to the same device as sample_tensor
        device = sample_tensor.device
        self.to(device)

        self._initialized = True
        print(f">> FeatureFusion initialized: method={self.method}, d_model={d_model}, num_features={num_features}")

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of tensors, each shape (B, S, D) and same D.
        Returns:
            fused tensor shape (B, S, D)
        """
        if not isinstance(features, (list, tuple)):
            raise ValueError("features must be a list/tuple of tensors")
        if len(features) == 0:
            raise ValueError("features list is empty")
        n = len(features)

        # basic shape checks
        B, S, D = features[0].shape
        for t in features:
            if t.shape != (B, S, D):
                raise ValueError("All feature tensors must have identical shape (B, S, D)")

        if not self._initialized:
            self._lazy_init(features[0], n)

        if self.method == "weighted":
            # scalar weight per feature (softmax over features)
            logits = self._weight_logits  # (n,)
            weights = F.softmax(logits, dim=0)  # (n,)
            # apply broadcasting: fused = sum_i w_i * feature_i
            fused = sum(w * f for w, f in zip(weights, features))
        elif self.method == "gated":
            # elementwise gates per feature: gate = sigmoid(Linear(feature))
            gated = []
            for lin, f in zip(self._gates, features):
                g = torch.sigmoid(lin(f))   # (B, S, D)
                gated.append(g * f)
            fused = sum(gated)
        elif self.method == "concat_linear":
            concat = torch.cat(features, dim=-1)  # (B, S, n*D)
            fused = self._concat_proj(concat)
        elif self.method == "self_attn":
            # 1) concat along sequence dimension: (B, S*n, D)
            cat_seq = torch.cat(features, dim=1)
            attn_out, _ = self._attn(cat_seq, cat_seq, cat_seq)  # (B, S*n, D)
            # reshape to (B, n, S, D)
            attn_out = attn_out.view(B, n, S, D)
            # average across features dimension -> (B, S, D)
            fused = attn_out.mean(dim=1)
        else:
            raise RuntimeError("Unsupported method")

        # optional final projection, dropout, layernorm and residual
        fused = self.out_proj(fused)
        fused = self.dropout(fused)
        if self.use_layernorm:
            fused = self.ln(fused)
        if self.residual:
            # add first feature as residual (requires same shape)
            fused = fused + features[0]
        return fused


class FFNFeatureFusion(nn.Module):
    def __init__(self, d_model: int, num_features: int, dropout: float = 0.1):
        super(FFNFeatureFusion, self).__init__()
        self.d_model = d_model
        self.num_features = num_features

        """
        factor = 3 / 4  # factor for intermediate layer size
        self.w_1 = nn.Linear(num_features * d_model, int((num_features * d_model) * factor))
        # tanh activation
        self.act_1 = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(int((num_features * d_model) * factor), d_model)
        """
        self.projector = nn.Linear(num_features * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of tensors, each shape (B, S, D) and same D.
        Returns:
            fused tensor shape (B, S, D)
        """
        if not isinstance(features, (list, tuple)):
            raise ValueError("features must be a list/tuple of tensors")
        if len(features) == 0:
            raise ValueError("features list is empty")
        
        # Concatenate features along the last dimension
        concatenated = torch.cat(features, dim=-1) # (B, S, num_features * d_model)
        
        """
        # Apply the first linear layer, activation, and dropout
        x = self.w_1(concatenated)
        x = self.act_1(x)
        x = self.dropout(x)
        # Apply the second linear layer
        x = self.w_2(x)
        # Sum the features if residual connection is needed
        if self.num_features == 2:
            x = x + (features[0] + features[1])
        elif self.num_features == 4:
            x = x + (features[0] + features[1] + features[2] + features[3])
        else:
            raise ValueError(f"[FFNFeatureFusion.forward] Unsupported num_features: {self.num_features}")
        """
        
        x = self.projector(concatenated) # (B, S, d_model)
        x = self.norm(x)    # Apply LayerNorm
        x = self.dropout(x) # Apply dropout
        
        # Return the fused features
        return x


class FeatEmbedding(nn.Module):

    def __init__(self, d_feat, d_model, dropout):
        super(FeatEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            nn.LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_model))

    def forward(self, x):
        return self.video_embeddings(x)


class TextEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb


def self_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # mask的操作在QK之后，softmax之前
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn, value), self_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            # 再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第一维添加一维，与后面的self attention计算维度一样
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)
        # if self.head == 1:
        #     x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        # else:
        #     query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]
        #     key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        #     value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        #
        #     x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        #     # 变为三维， 或者说是concat head
        #     x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]

        x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        # 变为三维， 或者说是concat head
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

        return self.linear_out(x)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout(self.relu(self.w_1(x)))
        output = self.w_2(inter)
        return output


# class SwiGLU(nn.Module):
#     """
#     A standard SwiGLU FFN implementation.
#     Reference: Noam Shazeer's "GLU Variants Improve Transformer"
#     (https://arxiv.org/abs/2002.05202)
#     """
#     def __init__(self, d_model: int, d_ff: int, multiple_of: int, dropout: float):
#         super(SwiGLU, self).__init__()
#         # Adjust hidden_dim to be a multiple of multiple_of
#         hidden_dim = multiple_of * ((2 * d_ff // 3 + multiple_of - 1) // multiple_of)
#         self.w1 = nn.Linear(d_model, hidden_dim)
#         self.act1 = nn.Sigmoid()
#         self.w2 = nn.Linear(d_model, hidden_dim)
#         self.w3 = nn.Linear(hidden_dim, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Forward pass using Swish activation and dropout
#         # return self.w3(self.dropout(F.tanh(self.w1(x)) * self.w2(x)))
#         inter1 = self.w1(x)
#         inter1 = inter1 * self.act1(inter1)  # Swish activation
#         inter2 = self.w2(x)
#         output = self.dropout(self.w3(inter1 * inter2))
#         return output


class MySwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(MySwiGLU, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inter1 = self.dropout1(F.silu(self.w1(x)))
        inter2 = self.dropout2(F.relu(self.w2(x)))
        output = self.w3(inter1 * inter2)
        return output
    

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm_1 = nn.LayerNorm(size)
        self.norm_2 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # return self.dropout(self.layer_norm(x + sublayer(x)))
        return self.dropout(x + self.norm_2(sublayer(self.norm_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x, self.feed_forward)


class EncoderLayerNoAttention(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayerNoAttention, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        return self.sublayer_connection[1](x, self.feed_forward)


class DecoderLayer(nn.Module):

    def __init__(self, size, attn, feed_forward, sublayer_num, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), sublayer_num)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, trg_mask))
        x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory, memory, src_mask))

        if r2l_memory is not None:
            x = self.sublayer_connection[-2](x, lambda x: self.attn(x, r2l_memory, r2l_memory, r2l_trg_mask))

        return self.sublayer_connection[-1](x, self.feed_forward)


class Encoder(nn.Module):

    def __init__(self, n, encoder_layer, d_model):
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer, n)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        x = self.norm_1(x)
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        x = self.norm_2(x)
        return x


class R2L_Decoder(nn.Module):

    def __init__(self, n, decoder_layer, d_model):
        super(R2L_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, r2l_trg_mask):
        x = self.norm_1(x)
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, r2l_trg_mask)
        x = self.norm_2(x)
        return x


class L2R_Decoder(nn.Module):

    def __init__(self, n, decoder_layer, d_model):
        super(L2R_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        x = self.norm_1(x)
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
        x = self.norm_2(x)
        return x


def pad_mask(src, r2l_trg, trg, pad_idx):
    if isinstance(src, tuple):
        if len(src) == 4:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
            src_rel_mask = (src[3][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask, src_rel_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            src_mask = (enc_src_mask, dec_src_mask)
        if len(src) == 3:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            src_mask = (enc_src_mask, dec_src_mask)
        if len(src) == 2:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            src_mask = (enc_src_mask, dec_src_mask)
    else:
        src_mask = (src[:, :, 0] != pad_idx).unsqueeze(1)
    if trg is not None:
        if isinstance(src_mask, tuple):
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(src_image_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_image_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_image_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
        else:
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(src_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask  # src_mask[batch, 1, lens]  trg_mask[batch, 1, lens]

    else:
        return src_mask


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(mask) == 0).cuda()


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class ABDTransformer(nn.Module):

    def __init__(self, vocab, d_feat, d_model, d_ff, n_heads, n_layers, dropout, feature_mode,
                 device='cuda', n_heads_big=128):
        super(ABDTransformer, self).__init__()
        self.vocab = vocab
        self.device = device
        self.feature_mode = feature_mode

        c = copy.deepcopy

        # attn_no_heads = MultiHeadAttention(1, d_model, dropout)

        attn = MultiHeadAttention(n_heads, d_model, dropout)

        attn_big = MultiHeadAttention(n_heads_big, d_model, dropout)

        # attn_big2 = MultiHeadAttention(10, d_model, dropout)

        # feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # feed_forward = SwiGLU(d_model=d_model, d_ff=d_ff, multiple_of=256, dropout=dropout)
        feed_forward = MySwiGLU(d_model=d_model, d_ff=d_ff, dropout=dropout)

        if feature_mode == 'one':
            self.src_embed = FeatEmbedding(d_feat, d_model, dropout)
        elif feature_mode == 'two':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
        elif feature_mode == 'three':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
        elif feature_mode == 'four':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
            self.rel_src_embed = FeatEmbedding(d_feat[3], d_model, dropout)
        self.trg_embed = TextEmbedding(vocab.n_vocabs, d_model)
        self.pos_embed = PositionalEncoding(d_model, dropout)

        # Feature fusion module
        """
        # choose from {"weighted", "gated", "concat_linear", "self_attn"}
        feat_fusion_method = "self_attn"
        self.feat_fusion = FeatureFusion(d_model=d_model, method=feat_fusion_method,
                                         num_heads=n_heads, dropout=dropout,
                                         use_layernorm=False, residual=False)
        """
        self.r2l_feat_fusion = FFNFeatureFusion(d_model=d_model, num_features=2, dropout=dropout)
        self.l2r_feat_fusion = FFNFeatureFusion(d_model=d_model, num_features=4, dropout=dropout)
        
        # self.encoder_no_heads = Encoder(n_layers, EncoderLayer(d_model, c(attn_no_heads), c(feed_forward), dropout))

        # self.encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout), d_model)
        self.img_encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout), d_model)
        self.mot_encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout), d_model)
        self.obj_encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout), d_model)

        # self.encoder_big = Encoder(n_layers, EncoderLayer(d_model, c(attn_big), c(feed_forward), dropout), d_model)
        self.img_encoder_big = Encoder(n_layers, EncoderLayer(d_model, c(attn_big), c(feed_forward), dropout), d_model)
        self.mot_encoder_big = Encoder(n_layers, EncoderLayer(d_model, c(attn_big), c(feed_forward), dropout), d_model)

        # self.encoder_big2 = Encoder(n_layers, EncoderLayer(d_model, c(attn_big2), c(feed_forward), dropout))

        # self.encoder_no_attention = Encoder(n_layers,EncoderLayerNoAttention(d_model, c(attn), c(feed_forward), dropout), d_model)
        self.rel_encoder_no_attention = Encoder(n_layers,EncoderLayerNoAttention(d_model, c(attn), c(feed_forward), dropout), d_model)

        self.r2l_decoder = R2L_Decoder(n_layers, 
                                       DecoderLayer(d_model, c(attn), c(feed_forward), sublayer_num=3, dropout=dropout),
                                       d_model)
        self.l2r_decoder = L2R_Decoder(n_layers, 
                                       DecoderLayer(d_model, c(attn), c(feed_forward), sublayer_num=4, dropout=dropout),
                                       d_model)

        # self.generator = Generator(d_model, vocab.n_vocabs)
        self.r2l_generator = Generator(d_model, vocab.n_vocabs)
        self.l2r_generator = Generator(d_model, vocab.n_vocabs)

    def encode(self, src, src_mask, feature_mode_two=False):
        # ============== Spatial-Temporal Encoding ==============
        if feature_mode_two:
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            # x1 = self.encoder_big(x1, src_mask[0])
            x1 = self.img_encoder_big(x1, src_mask[0])
            
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            # x2 = self.encoder_big(x2, src_mask[1])
            x2 = self.mot_encoder_big(x2, src_mask[1])
            
            # return x1 + x2
            return self.r2l_feat_fusion([x1, x2])
        
        # ============== Object-Relation Encoding ==============
        if self.feature_mode == 'one':
            raise NotImplementedError("[ABDTransformer.encode] Feature mode 'one' is not supported for encoding.")
            x = self.src_embed(src)
            x = self.pos_embed(x)
            return self.encoder(x, src_mask)
        elif self.feature_mode == 'two':
            raise NotImplementedError("[ABDTransformer.encode] Feature mode 'two' is not supported for encoding.")
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, src_mask[1])
            return x1 + x2
        elif self.feature_mode == 'three':
            raise NotImplementedError("[ABDTransformer.encode] Feature mode 'three' is not supported for encoding.")
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])
            x3 = self.object_src_embed(src[2])
            x3 = self.pos_embed(x3)
            x3 = self.encoder(x3, src_mask[2])
            return x1 + x2 + x3
        elif self.feature_mode == 'four':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            # x1 = self.encoder(x1, src_mask[0])
            x1 = self.img_encoder(x1, src_mask[0])

            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            # x2 = self.encoder(x2, src_mask[1])
            x2 = self.mot_encoder(x2, src_mask[1])

            x3 = self.object_src_embed(src[2])
            # x3 = self.pos_embed(x3)
            # x3 = self.encoder(x3, src_mask[2])
            x3 = self.obj_encoder(x3, src_mask[2])
            # x3 = self.encoder_no_attention(x3, src_mask[2])

            x4 = self.rel_src_embed(src[3])
            # x4 = self.pos_embed(x4)
            # x4 = self.encoder(x4, src_mask[3])
            # x4 = self.encoder_no_attention(x4, src_mask[3])
            x4 = self.rel_encoder_no_attention(x4, src_mask[3])
            
            # return x1 + x2 + x3 + x4
            # return self.feat_fusion([x1, x2, x3, x4])
            return self.l2r_feat_fusion([x1, x2, x3, x4])

    def r2l_decode(self, r2l_trg, memory, src_mask, r2l_trg_mask):
        x = self.trg_embed(r2l_trg)
        x = self.pos_embed(x)
        return self.r2l_decoder(x, memory, src_mask, r2l_trg_mask)

    def l2r_decode(self, trg, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        x = self.trg_embed(trg)
        x = self.pos_embed(x)
        return self.l2r_decoder(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)

    def forward(self, src, r2l_trg, trg, mask):
        src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask = mask
        if self.feature_mode == 'one':
            encoding_outputs = self.encode(src, src_mask)
            r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, src_mask, r2l_trg_mask)
            l2r_outputs = self.l2r_decode(trg, encoding_outputs, src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

        elif self.feature_mode == 'two' or 'three' or 'four':
            enc_src_mask, dec_src_mask = src_mask
            r2l_encoding_outputs = self.encode(src, enc_src_mask, feature_mode_two=True)
            encoding_outputs = self.encode(src, enc_src_mask)

            r2l_outputs = self.r2l_decode(r2l_trg, r2l_encoding_outputs, dec_src_mask, r2l_trg_mask)
            l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

            # r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, dec_src_mask, r2l_trg_mask)
            # l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, None, None)
        else:
            raise "没有输出"

        # r2l_pred = self.generator(r2l_outputs)
        # l2r_pred = self.generator(l2r_outputs)
        r2l_pred = self.r2l_generator(r2l_outputs)
        l2r_pred = self.l2r_generator(l2r_outputs)
        
        return r2l_pred, l2r_pred

    def greedy_decode(self, batch_size, src_mask, memory, max_len):
        raise NotImplementedError(f"[ABDTransformer.greedy_decode] Method not implemented. Use beam search instead.")
        eos_idx = self.vocab.word2idx['<S>']
        r2l_hidden = None
        with torch.no_grad():
            output = torch.ones(batch_size, 1).fill_(eos_idx).long().cuda()
            for i in range(max_len + 2 - 1):
                trg_mask = subsequent_mask(output.size(1))
                dec_out = self.r2l_decode(output, memory, src_mask, trg_mask)  # batch, len, d_model
                r2l_hidden = dec_out
                pred = self.generator(dec_out)  # batch, len, n_vocabs
                next_word = pred[:, -1].max(dim=-1)[1].unsqueeze(1)  # pred[:, -1]([batch, n_vocabs])
                output = torch.cat([output, next_word], dim=-1)
        return r2l_hidden, output

    def r2l_beam_search_decode(self, batch_size, src, src_mask, model_encodings, beam_size, max_len):
        end_symbol = self.vocab.word2idx['<S>']
        start_symbol = self.vocab.word2idx['<S>']

        r2l_outputs = None

        # 1.1 Setup Src
        "src has shape (batch_size, sent_len)"
        "src_mask has shape (batch_size, 1, sent_len)"
        # src_mask = (src[:, :, 0] != self.vocab.word2idx['<PAD>']).unsqueeze(-2)  # TODO Untested
        "model_encodings has shape (batch_size, sentence_len, d_model)"
        # model_encodings = self.encode(src, src_mask)

        # 1.2 Setup Tgt Hypothesis Tracking
        "hypothesis is List(4 bt)[(cur beam_sz, dec_sent_len)], init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]  # probs are log_probs must be init at 0.

        # 2. Iterate: Generate one char at a time until maxlen
        for iter in range(max_len + 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break

            # 2.1 Setup the batch. Since we use beam search, each batch has a variable number (called cur_beam_size)
            # between 0 and beam_size of hypotheses live at any moment. We decode all hypotheses for all batches at
            # the same time, so we must copy the src_encodings, src_mask, etc the appropriate number fo times for
            # the number of hypotheses for each example. We keep track of the number of live hypotheses for each example.
            # We run all hypotheses for all examples together through the decoder and log-softmax,
            # and then use `torch.split` to get the appropriate number of hypotheses for each example in the end.
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l = [], [], [], []
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [model_encodings[i:i + 1]] * cur_beam_size
                src_mask_l += [src_mask[i:i + 1]] * cur_beam_size
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            if self.feature_mode == 'one':
                out = self.r2l_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src.data)).to(self.device))
            elif self.feature_mode == 'two' or 'three' or 'four':
                out = self.r2l_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[0].data)).to(self.device))
            r2l_outputs = out

            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            # log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            log_prob = self.r2l_generator(out[:, -1, :]).unsqueeze(1)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            _, decoded_len, vocab_sz = log_prob.shape
            # log_prob = log_prob.reshape(batch_size, cur_beam_size, decoded_len, vocab_sz)
            "shape List(4 bt)[(cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)]"
            "log_prob[i] is (cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)"
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)

            # 2.2 Now we process each example in the batch. Note that the example may have already finished processing before
            # other examples (no more hypotheses to try), in which case we continue
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                # 2.2.1 We compute the cumulative scores for each live hypotheses for the example
                # hyp_scores is the old scores for the previous stage, and `log_prob` are the new probs for
                # this stage. Since they are log probs, we sum them instaed of multiplying them.
                # The .view(-1) forces all the hypotheses into one dimension. The shape of this dimension is
                # cur_beam_sz * vocab_sz (ex: 5 * 50002). So after getting the topk from it, we can recover the
                # generating sentence and the next word using: ix // vocab_sz, ix % vocab_sz.
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                "shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)

                # 2.2.2 We get the topk values in cumulative_hyp_scores_i and compute the current (generating) sentence
                # and the next word using: ix // vocab_sz, ix % vocab_sz.
                "shape (cur_beam_sz,)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                "shape (cur_beam_sz,). Vals are between 0 and 50002 vocab_sz"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                "shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs, \
                                             top_cand_hyp_pos % self.vocab.n_vocabs

                # 2.2.3 For each of the topk words, we append the new word to the current (generating) sentence
                # We add this to new_hypotheses_i and add its corresponding total score to new_hyp_scores_i
                new_hypotheses_i, new_hyp_scores_i = [], []  # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids,
                                                                        top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()

                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id], device=self.device)))
                    if hyp_word_id == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)

                # 2.2.4 We may find that the hypotheses_i for some example in the batch
                # is empty - we have fully processed that example. We use None as a sentinel in this case.
                # Above, the loops gracefully handle None examples.
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            # print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        # 2.3 Finally, we do some postprocessing to get our final generated candidate sentences.
        # Sometimes, we may get to max_len of a sentence and still not generate the </s> end token.
        # In this case, the partial sentence we have generated will not be added to the completed_hypotheses
        # automatically, and we have to manually add it in. We add in as many as necessary so that there are
        # `beam_size` completed hypotheses for each example.
        # Finally, we sort each completed hypothesis by score.
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        return r2l_outputs, completed_hypotheses

    def beam_search_decode(self, src, beam_size, max_len):
        """
                An Implementation of Beam Search for the Transformer Model.
                Beam search is performed in a batched manner. Each example in a batch generates `beam_size` hypotheses.
                We return a list (len: batch_size) of list (len: beam_size) of Hypothesis, which contain our output decoded sentences
                and their scores.
                :param src: shape (sent_len, batch_size). Each val is 0 < val < len(vocab_dec). The input tokens to the decoder.
                :param max_len: the maximum length to decode
                :param beam_size: the beam size to use
                :return completed_hypotheses: A List of length batch_size, each containing a List of beam_size Hypothesis objects.
                    Hypothesis is a named Tuple, its first entry is "value" and is a List of strings which contains the translated word
                    (one string is one word token). The second entry is "score" and it is the log-prob score for this translated sentence.
                Note: Below I note "4 bt", "5 beam_size" as the shapes of objects. 4, 5 are default values. Actual values may differ.
                """
        # 1. Setup
        start_symbol = self.vocab.word2idx['<S>']
        end_symbol = self.vocab.word2idx['<S>']

        # 1.1 Setup Src
        "src has shape (batch_size, sent_len)"
        "src_mask has shape (batch_size, 1, sent_len)"
        # src_mask = (src[:, :, 0] != self.vocab.word2idx['<PAD>']).unsqueeze(-2)  # TODO Untested
        src_mask = pad_mask(src, r2l_trg=None, trg=None, pad_idx=self.vocab.word2idx['<PAD>'])
        "model_encodings has shape (batch_size, sentence_len, d_model)"
        if self.feature_mode == 'one':
            batch_size = src.shape[0]
            model_encodings = self.encode(src, src_mask)
            r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, src_mask,
                                                                               model_encodings=model_encodings,
                                                                               beam_size=beam_size, max_len=max_len)
        elif self.feature_mode == 'two' or 'three' or 'four':
            batch_size = src[0].shape[0]
            enc_src_mask = src_mask[0]
            dec_src_mask = src_mask[1]
            r2l_model_encodings = self.encode(src, enc_src_mask, feature_mode_two=True)
            # model_encodings = r2l_model_encodings
            model_encodings = self.encode(src, enc_src_mask)

            r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, dec_src_mask,
                                                                               model_encodings=r2l_model_encodings,
                                                                               beam_size=beam_size, max_len=max_len)

        # 1.2 Setup r2l target output
        # r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, src_mask,
        #                                                                    model_encodings=model_encodings,
        #                                                                    beam_size=1, max_len=max_len)
        # r2l_memory, r2l_completed_hypotheses = self.greedy_decode(batch_size, src_mask, model_encodings, max_len)
        # beam_r2l_memory = [copy.deepcopy(r2l_memory) for _ in range(beam_size)]
        # 1.3 Setup Tgt Hypothesis Tracking
        "hypothesis is List(4 bt)[(cur beam_sz, dec_sent_len)], init: List(4 bt)[(1 init_beam_sz, dec_sent_len)]"
        "hypotheses[i] is shape (cur beam_sz, dec_sent_len)"
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long,
                                               device=self.device)) for _ in range(batch_size)]
        "List after init: List 4 bt of List of len max_len_completed, init: List of len 4 bt of []"
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        "List len batch_sz of shape (cur beam_sz), init: List(4 bt)[(1 init_beam_sz)]"
        "hyp_scores[i] is shape (cur beam_sz)"
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.float, device=self.device))
                      for _ in range(batch_size)]  # probs are log_probs must be init at 0.

        # 2. Iterate: Generate one char at a time until maxlen
        for iter in range(max_len + 1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]):
                break

            # 2.1 Setup the batch. Since we use beam search, each batch has a variable number (called cur_beam_size)
            # between 0 and beam_size of hypotheses live at any moment. We decode all hypotheses for all batches at
            # the same time, so we must copy the src_encodings, src_mask, etc the appropriate number fo times for
            # the number of hypotheses for each example. We keep track of the number of live hypotheses for each example.
            # We run all hypotheses for all examples together through the decoder and log-softmax,
            # and then use `torch.split` to get the appropriate number of hypotheses for each example in the end.
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l, r2l_memory_l = [], [], [], [], []
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [model_encodings[i:i + 1]] * cur_beam_size
                if self.feature_mode == 'one':
                    src_mask_l += [src_mask[i:i + 1]] * cur_beam_size
                elif self.feature_mode == 'two' or 'three' or 'four':
                    src_mask_l += [dec_src_mask[i:i + 1]] * cur_beam_size
                r2l_memory_l += [r2l_memory[i: i + 1]] * cur_beam_size
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            r2l_memory_cur = torch.cat(r2l_memory_l, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            if self.feature_mode == 'one':
                out = self.l2r_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src.data)).to(self.device),
                                      r2l_memory_cur, r2l_trg_mask=None)
            elif self.feature_mode == 'two' or 'three' or 'four':
                out = self.l2r_decode(Variable(y_tm1).to(self.device), model_encodings_cur, src_mask_cur,
                                      Variable(subsequent_mask(y_tm1.size(-1)).type_as(src[0].data)).to(self.device),
                                      r2l_memory_cur, r2l_trg_mask=None)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            # log_prob = self.generator(out[:, -1, :]).unsqueeze(1)
            log_prob = self.l2r_generator(out[:, -1, :]).unsqueeze(1)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 50002 vocab_sz)"
            _, decoded_len, vocab_sz = log_prob.shape
            # log_prob = log_prob.reshape(batch_size, cur_beam_size, decoded_len, vocab_sz)
            "shape List(4 bt)[(cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)]"
            "log_prob[i] is (cur_beam_sz_i, dec_sent_len, 50002 vocab_sz)"
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)

            # 2.2 Now we process each example in the batch. Note that the example may have already finished processing before
            # other examples (no more hypotheses to try), in which case we continue
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue

                # 2.2.1 We compute the cumulative scores for each live hypotheses for the example
                # hyp_scores is the old scores for the previous stage, and `log_prob` are the new probs for
                # this stage. Since they are log probs, we sum them instaed of multiplying them.
                # The .view(-1) forces all the hypotheses into one dimension. The shape of this dimension is
                # cur_beam_sz * vocab_sz (ex: 5 * 50002). So after getting the topk from it, we can recover the
                # generating sentence and the next word using: ix // vocab_sz, ix % vocab_sz.
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                "shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1)
                                           .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i]).view(-1)

                # 2.2.2 We get the topk values in cumulative_hyp_scores_i and compute the current (generating) sentence
                # and the next word using: ix // vocab_sz, ix % vocab_sz.
                "shape (cur_beam_sz,)"
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                "shape (cur_beam_sz,). Vals are between 0 and 50002 vocab_sz"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                "shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.vocab.n_vocabs, \
                                             top_cand_hyp_pos % self.vocab.n_vocabs

                # 2.2.3 For each of the topk words, we append the new word to the current (generating) sentence
                # We add this to new_hypotheses_i and add its corresponding total score to new_hyp_scores_i
                new_hypotheses_i, new_hyp_scores_i = [], []  # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids,
                                                                        top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()

                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id], device=self.device)))
                    if hyp_word_id == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)

                # 2.2.4 We may find that the hypotheses_i for some example in the batch
                # is empty - we have fully processed that example. We use None as a sentinel in this case.
                # Above, the loops gracefully handle None examples.
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0, -1).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            # print(new_hypotheses, new_hyp_scores)
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores

        # 2.3 Finally, we do some postprocessing to get our final generated candidate sentences.
        # Sometimes, we may get to max_len of a sentence and still not generate the </s> end token.
        # In this case, the partial sentence we have generated will not be added to the completed_hypotheses
        # automatically, and we have to manually add it in. We add in as many as necessary so that there are
        # `beam_size` completed hypotheses for each example.
        # Finally, we sort each completed hypothesis by score.
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                        value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id][1:]],
                        score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
        # print('completed_hypotheses', completed_hypotheses)
        return r2l_completed_hypotheses, completed_hypotheses
