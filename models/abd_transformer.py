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


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class FFNFeatureFusion(nn.Module):
    def __init__(self, d_model: int, num_features: int, dropout: float):
        super(FFNFeatureFusion, self).__init__()
        lowrank_dim = max(128, d_model // 4)
        self.bottlenecks = nn.ModuleList([nn.Linear(d_model, lowrank_dim) for _ in range(num_features)])
        self.lns = nn.ModuleList([nn.LayerNorm(lowrank_dim) for _ in range(num_features)])
        self.dropout = nn.Dropout(dropout)
        self.activation = NewGELUActivation()
        # final projector from concat(lowrank) -> d_model
        self.final = nn.Linear(num_features * lowrank_dim, d_model)
        self.out_ln = nn.LayerNorm(d_model)
        self.num_features = num_features
        
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
        if len(features) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {len(features)}")
        
        proj = []
        for f, proj_layer, ln in zip(features, self.bottlenecks, self.lns):
            z = proj_layer(f)             # (B,S,bottleneck) linear
            z = self.activation(z)        # non-linear (GELU/ReLU)
            z = ln(z)                     # LayerNorm on last dim
            z = self.dropout(z)
            proj.append(z)
        x = torch.cat(proj, dim=-1)       # (B,S, num*bottleneck)
        out = self.final(x)               # (B,S,D)
        out = self.out_ln(out)
        return out


class FeatEmbedding(nn.Module):

    def __init__(self, d_feat, d_model, dropout):
        super(FeatEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            nn.LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_model)
        )

    def forward(self, x):
        x = self.video_embeddings(x)
        return x


class TextEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


def apply_rotary_positional_embedding(
    x: torch.Tensor, theta_is: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary positional embedding to input tensor x.

    Args:
        x : Input tensor of shape (B, T, D).
        theta_is : Precomputed complex frequencies of shape (T, D//2).

    Returns:
        torch.Tensor: Tensor with RoPE applied, same shape as input x.
    """

    # Reshape x to separate real and imaginary parts (pairs of features)
    # Shape: (B, T, D)    -> (B, T, D//2, 2)
    x_combined = x.float().reshape(*x.shape[:-1], -1, 2)

    # Convert to complex numbers
    # Shape: (B, T, D//2)
    x_complex = torch.view_as_complex(x_combined)

    # Reshape theta_is for broadcasting
    # Shape: (T, D//2) -> (1, T, D//2)
    # Add batch dimension (dim=0)
    theta_is = theta_is.unsqueeze(0)

    # Apply rotation by multiplying complex numbers
    # Broadcasting works:
    # (B, T, D//2) * (1, T, D//2) -> (B, T, D//2)
    x_rotated = x_complex * theta_is.to(x_complex.device)

    # Convert back to real numbers
    # Shape: (B, T, D//2, 2)
    x_out = torch.view_as_real(x_rotated)

    # Reshape back to original input shape
    # Shape: (B, T, D)
    # Flatten the last two dimensions (D//2, 2) -> D
    x_out = x_out.flatten(start_dim=-2)
    return x_out.type_as(x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        n_embd: int,
        block_size: int,
        base_frequency: int = 10000,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initializes the Rotary Positional Embedding.

        Args:
            n_embd : Dimension of the head embeddings. Must be even.
            block_size : Maximum sequence length.
            base_frequency : The base_frequency value for frequency calculation (theta_i).
            device : Device to store the precomputed frequencies.
        """
        super(RotaryPositionalEmbedding, self).__init__()
        if n_embd % 2 != 0:
            raise ValueError("Dimension must be even for RoPE.")

        self.n_embd = n_embd
        self.block_size = block_size
        self.base_frequency = base_frequency
        self.device = device

        # Precompute theta values for RoPE
        # For each i in [0, n_embd/ 2 ):
        #   theta_i = 1 / (base_frequency^(2 * i / n_embd))
        half_dimension = self.n_embd // 2
        frequency_range = torch.arange(half_dimension, dtype=torch.float32)
        exponent = 2 * frequency_range / self.n_embd
        denominator = torch.pow(self.base_frequency, exponent)
        # Shape: (n_embd // 2,)
        theta_is = 1.0 / denominator

        # Calculate frequencies for each position: m * theta
        # Shape: (block_size, n_embd / 2)
        position_indices = torch.arange(self.block_size, dtype=torch.float32)
        frequencies = torch.outer(position_indices, theta_is)

        # Calculate complex numbers in polar form: cos(m*theta) + i*sin(m*theta)
        # Shape: (block_size, n_embd / 2)
        self.theta_is = torch.polar(torch.ones_like(frequencies), frequencies)

        if self.device:
            self.theta_is = self.theta_is.to(self.device)

    def get_theta_is(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """
        Returns the precomputed complex frequencies for a given sequence length.

        Args:
            sequence_length : The sequence length (T).
            device : Target device.

        Returns:
            Complex frequencies of shape (sequence_length, n_embd / 2).
        """
        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds maximum precomputed length {self.block_size}"
            )

        # Return the slice for the current sequence length T
        self.theta_is = self.theta_is.to(device)
        return self.theta_is[:sequence_length]


class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)
        assert self.pe.ndim == 2
        
        self.drop_out = nn.Dropout(p=dropout)
        #self.dim = dim

    def forward(self, emb, step=None):

        #emb = emb * math.sqrt(self.dim)'
        if step is None:
            emb = emb + self.pe[:emb.size(1), :]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention function.
    Inputs: query, key, value [batch_size, num_heads, seq_len, d_k]
    mask: [batch_size, 1, seq_q, seq_k] or broadcastable
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # Assuming mask is 0/1, 0 to mask out
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention for batch_first: [batch_size, seq_len, d_model].
    Can be used for encoder (self-attn on video features) or decoder (self-attn on captions, cross-attn to video).
    
    Multi-Head Latent Attention (MLA) from DeepSeek-V2, simplified without RoPE.
    """
    def __init__(self, num_heads: int, d_model: int, dropout: float, use_rope: bool) -> None:
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads (got {d_model} % {num_heads} != 0)")
        if use_rope:
            raise ValueError("Rotary Positional Embedding (RoPE) is not supported in this implementation of MultiHeadAttention.")
        
        # --- MLA Parameters ---
        n_embd = d_model
        kv_compression_dim = max(128, d_model // 4)
        q_compression_dim = d_model // 2
        self.head_size = n_embd // num_heads
        self.rotary_embeddings = RotaryPositionalEmbedding(
            n_embd=n_embd,
            block_size=1_000, # d_model * 2
            base_frequency=10_000,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # ----------------------
        
        # --- Query Compression Path ---
        # Down-projection for query
        self.W_dq = nn.Linear(n_embd, q_compression_dim)
        # LayerNorm after query down-projection
        self.q_layer_norm = nn.LayerNorm(q_compression_dim)
        # Up-projection for query
        self.W_uq = nn.Linear(q_compression_dim, n_embd)
        
        # --- Key-Value Joint Compression Path ---
        # Down-projection for key-value
        self.W_dkv = nn.Linear(n_embd, kv_compression_dim)
        # LayerNorm after key-value down-projection
        self.kv_layer_norm = nn.LayerNorm(kv_compression_dim)
        # Up-projection for key (from compressed KV)
        self.W_uk = nn.Linear(kv_compression_dim, n_embd)
        # Up-projection for value (from compressed KV)
        self.W_uv = nn.Linear(kv_compression_dim, n_embd)
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.activation = NewGELUActivation()
        
        self.output_linear = nn.Linear(d_model, d_model)
        """
        self.output_linear = nn.Sequential(
            nn.Linear(d_model, d_model // 3),
            NewGELUActivation(),
            nn.Linear(d_model // 3, d_model)
        )
        """
        self.dropout = nn.Dropout(p=dropout)
        self.use_rope = use_rope
        self.attn_weights = None  # To store if needed

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: [batch_size, seq_len, d_model]
        mask: Optional [batch_size, seq_q, seq_k] with 0/1 (0 to mask)
        """

        # Batch size, Sequence length, Embedding dimensionality
        B, T, C = query.shape

        # 1. Query Path
        # (B, T, C) -> (B, T, q_compression_dim)
        compressed_q_latent = self.activation(self.W_dq(query))
        compressed_q_latent_norm = self.q_layer_norm(compressed_q_latent)
        # (B, T, q_compression_dim) -> (B, T, C)
        q_final = self.W_uq(compressed_q_latent_norm)

        # 2. Key-Value Path
        # (B, T, C) -> (B, T, kv_compression_dim)
        compressed_kv_latent = self.activation(self.W_dkv(key))
        compressed_kv_latent_norm = self.kv_layer_norm(compressed_kv_latent)
        # (B, T, kv_compression_dim) -> (B, T, C)
        k_final = self.W_uk(compressed_kv_latent_norm)
        # (B, T, kv_compression_dim) -> (B, T, C)
        v_final = self.W_uv(compressed_kv_latent_norm)

        # 3. Apply Rotary Positional Embedding to Q, K
        if self.use_rope:
            theta_is = self.rotary_embeddings.get_theta_is(q_final.shape[1], query.device)
            q_rope = apply_rotary_positional_embedding(q_final, theta_is)
            theta_is = self.rotary_embeddings.get_theta_is(k_final.shape[1], key.device)
            k_rope = apply_rotary_positional_embedding(k_final, theta_is)
        else:
            q_rope = q_final
            k_rope = k_final
        
        # 4. Reshape Q, K, V for multi-head attention
        # (B, T, C) -> (B, T, num_heads, head_size) -> (B, num_heads, T, head_size)
        q_heads = q_rope.view(B, -1, self.num_heads, self.head_size).transpose(1, 2)
        k_heads = k_rope.view(B, -1, self.num_heads, self.head_size).transpose(1, 2)
        v_heads = v_final.view(B, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        # Expand mask if provided: [batch_size, 1, seq_q, seq_k]
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dim for broadcasting
            mask = mask.to(query.device)
        
        # 5. Scaled Dot-Product Attention
        x, self.attn_weights = scaled_dot_product_attention(
            query=q_heads,
            key=k_heads,
            value=v_heads,
            mask=mask, 
            dropout=self.dropout
        )
        
        # 6. Concat heads: [batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        
        return self.output_linear(x)


class SwiGLU(nn.Module):
    """
    A standard SwiGLU FFN implementation.
    Reference: Noam Shazeer's "GLU Variants Improve Transformer"
    (https://arxiv.org/abs/2002.05202)
    """
    def __init__(self, d_model: int, d_ff: int, multiple_of: int, dropout: float):
        super(SwiGLU, self).__init__()
        # Adjust hidden_dim to be a multiple of multiple_of
        hidden_dim = multiple_of * ((2 * d_ff // 3 + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.main_activation = nn.SiLU()  # Swish activation
        self.sub_activation = NewGELUActivation()
        """
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass using Swish activation and dropout
        x1 = self.main_activation(self.w1(x))
        #x1 = self.sub_activation(self.w1(x))'
        x2 = self.w2(x)
        y = self.dropout(self.w3(x1 * x2))
        """
        y = self.transform(x)
        """
        return y


class SublayerConnection(nn.Module):

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm_1 = nn.LayerNorm(size)
        self.norm_2 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # return self.dropout(self.layer_norm(x + sublayer(x)))
        return self.dropout(x + self.norm_2(sublayer(self.norm_1(x))))


class SkipConnectionAfterLN(nn.Module):

    def __init__(self, size: int, dropout: float):
        super(SkipConnectionAfterLN, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(x + self.norm(sublayer(x)))
    

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, multiple_of: int, num_heads: int, dropout: float, use_rope: bool, first_layer: bool):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.feed_forward = SwiGLU(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, dropout=dropout)
        if first_layer:
            self.sublayer_connection_1 = SkipConnectionAfterLN(size=d_model, dropout=dropout)
        else:
            self.sublayer_connection_1 = SublayerConnection(size=d_model, dropout=dropout)
        self.sublayer_connection_2 = SublayerConnection(size=d_model, dropout=dropout)

    def forward(self, x, mask):
        x = self.sublayer_connection_1(x, lambda x: self.attn(x, x, x, mask))
        x = self.sublayer_connection_2(x, self.feed_forward)
        return x


class EncoderLayerNoAttention(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, multiple_of: int, dropout: float, first_layer: bool):
        super(EncoderLayerNoAttention, self).__init__()
        self.feed_forward = SwiGLU(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, dropout=dropout)
        if first_layer:
            self.sublayer_connection = SkipConnectionAfterLN(size=d_model, dropout=dropout)
        else:
            self.sublayer_connection = SublayerConnection(size=d_model, dropout=dropout)

    def forward(self, x, mask):
        x = self.sublayer_connection(x, self.feed_forward)
        return x


# class DecoderLayer(nn.Module):

#     def __init__(self, size, attn, feed_forward, sublayer_num, dropout):
#         super(DecoderLayer, self).__init__()
#         self.attn = attn
#         self.feed_forward = feed_forward
#         self.sublayer_connection = clones(SublayerConnection(size, dropout), sublayer_num)

#     def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
#         x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, trg_mask))
#         x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory, memory, src_mask))

#         if r2l_memory is not None:
#             x = self.sublayer_connection[-2](x, lambda x: self.attn(x, r2l_memory, r2l_memory, r2l_trg_mask))

#         return self.sublayer_connection[-1](x, self.feed_forward)


class R2LDecoderLayer(nn.Module):

    def __init__(self, d_model: int, d_ff: int, multiple_of: int, num_heads: int, sublayer_num: int, dropout: float, use_rope: bool, first_layer: bool):
        assert sublayer_num == 3, "[R2LDecoderLayer.__init__] sublayer_num must be 3"
        super(R2LDecoderLayer, self).__init__()
        self.attn_1 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.attn_2 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.feed_forward = SwiGLU(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, dropout=dropout)
        if first_layer:
            self.sublayer_connection_1 = SkipConnectionAfterLN(size=d_model, dropout=dropout)
        else:
            self.sublayer_connection_1 = SublayerConnection(size=d_model, dropout=dropout)
        self.sublayer_connection_2 = SublayerConnection(size=d_model, dropout=dropout)
        self.sublayer_connection_3 = SublayerConnection(size=d_model, dropout=dropout)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        if r2l_memory is not None:
            raise ValueError("[R2LDecoderLayer.forward] r2l_memory must be None")
        
        x = self.sublayer_connection_1(x, lambda x: self.attn_1(x, x, x, trg_mask))
        x = self.sublayer_connection_2(x, lambda x: self.attn_2(x, memory, memory, src_mask))
        x = self.sublayer_connection_3(x, self.feed_forward)
        return x


class L2RDecoderLayer(nn.Module):

    def __init__(self, d_model: int, d_ff: int, multiple_of: int, num_heads: int, sublayer_num: int, dropout: float, use_rope: bool, first_layer: bool):
        assert sublayer_num == 4, "[L2RDecoderLayer.__init__] sublayer_num must be 4"
        super(L2RDecoderLayer, self).__init__()
        self.attn_1 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.attn_2 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.attn_3 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.feed_forward = SwiGLU(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, dropout=dropout)
        if first_layer:
            self.sublayer_connection_1 = SkipConnectionAfterLN(size=d_model, dropout=dropout)
        else:
            self.sublayer_connection_1 = SublayerConnection(size=d_model, dropout=dropout)
        self.sublayer_connection_2 = SublayerConnection(size=d_model, dropout=dropout)
        self.sublayer_connection_3 = SublayerConnection(size=d_model, dropout=dropout)
        self.sublayer_connection_4 = SublayerConnection(size=d_model, dropout=dropout)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        if r2l_memory is None:
            raise ValueError("[L2RDecoderLayer.forward] r2l_memory must not be None")
            
        x = self.sublayer_connection_1(x, lambda x: self.attn_1(x, x, x, trg_mask))
        x = self.sublayer_connection_2(x, lambda x: self.attn_2(x, memory, memory, src_mask))
        x = self.sublayer_connection_3(x, lambda x: self.attn_3(x, r2l_memory, r2l_memory, r2l_trg_mask))
        x = self.sublayer_connection_4(x, self.feed_forward)
        return x


class Encoder(nn.Module):

    def __init__(self, d_model: int, d_ff: int, multiple_of: int, num_heads: int, num_layers: int, dropout: float, use_rope: bool):
        super(Encoder, self).__init__()
        self.in_norm = nn.LayerNorm(d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=num_heads, dropout=dropout, use_rope=use_rope,
                         first_layer=(i == 0))
            for i in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        x = self.in_norm(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        x = self.out_norm(x)
        return x


class R2L_Decoder(nn.Module):

    def __init__(self, d_model: int, d_ff: int, multiple_of: int, num_heads: int, num_layers: int, dropout: float, use_rope: bool):
        super(R2L_Decoder, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.decoder_layers = nn.ModuleList([
            R2LDecoderLayer(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=num_heads, sublayer_num=3, 
                            dropout=dropout, use_rope=use_rope, first_layer=(i == 0))
            for i in range(num_layers)
        ])

    def forward(self, x, memory, src_mask, r2l_trg_mask):
        x = self.norm_1(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, r2l_trg_mask)
        x = self.norm_2(x)
        return x


class L2R_Decoder(nn.Module):

    def __init__(self, d_model: int, d_ff: int, multiple_of: int, num_heads: int, num_layers: int, dropout: float, use_rope: bool):
        super(L2R_Decoder, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.decoder_layers = nn.ModuleList([
            L2RDecoderLayer(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=num_heads, sublayer_num=4, 
                            dropout=dropout, use_rope=use_rope, first_layer=(i == 0))
            for i in range(num_layers)
        ])

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        x = self.norm_1(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
        x = self.norm_2(x)
        return x


def pad_mask(src, r2l_trg, trg, pad_idx):
    if isinstance(src, tuple):
        if len(src) == 4:
            raise ValueError("[pad_mask] src should not be a tuple of length 4.")
        if len(src) == 3:
            m1 = (src[0][:, :, 0] != pad_idx)
            assert m1.ndim == 2, f"[pad_mask] m1.ndim={m1.ndim}"
            assert m1.shape[1] == 10, f"[pad_mask] m1.shape={m1.shape}"

            m2 = (src[1][:, :, 0] != pad_idx)
            assert m2.ndim == 2, f"[pad_mask] m2.ndim={m2.ndim}"
            assert m2.shape[1] == 10, f"[pad_mask] m2.shape={m2.shape}"

            m3 = (src[2][:, :, 0] != pad_idx)
            assert m3.ndim == 2, f"[pad_mask] m3.ndim={m3.ndim}"
            assert m3.shape[1] == 10, f"[pad_mask] m3.shape={m3.shape}"

            _stacked = torch.stack([m1, m2, m3], dim=2)
            out_mask = _stacked.reshape(m1.size(0), -1)
            assert out_mask.ndim == 2, f"[pad_mask] out_mask.ndim={out_mask.ndim}"
            assert out_mask.shape[1] == 30, f"[pad_mask] out_mask.shape={out_mask.shape}"
            assert torch.equal(out_mask[:,0::3], m1[:,:]) == True
            assert torch.equal(out_mask[:,1::3], m2[:,:]) == True
            assert torch.equal(out_mask[:,2::3], m3[:,:]) == True

            enc_src_mask = out_mask.unsqueeze(1)
            dec_src_mask = out_mask.unsqueeze(1)
            src_mask = (enc_src_mask, dec_src_mask)
        if len(src) == 2:
            raise ValueError("[pad_mask] src should not be a tuple of length 2.")
    else:
        src_mask = (src[:, :, 0] != pad_idx).unsqueeze(1)
    if trg is not None:
        if isinstance(src_mask, tuple):
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(m1.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(m1.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(m1.data)
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

    def __init__(self, vocab, d_feat, d_model, d_ff, n_heads, n_heads_big,
                 n_enc_layers, n_dec_layers, dropout, max_caption_len):
        super(ABDTransformer, self).__init__()
        self.vocab = vocab
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        multiple_of = 128

        self.r2l_image_src_embed  = FeatEmbedding(d_feat[0], d_model, dropout)
        self.r2l_motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
        self.r2l_object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
        
        self.l2r_image_src_embed  = FeatEmbedding(d_feat[0], d_model, dropout)
        self.l2r_motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
        self.l2r_object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
        
        self.r2l_trg_embed = TextEmbedding(vocab.n_vocabs, d_model)
        self.l2r_trg_embed = TextEmbedding(vocab.n_vocabs, d_model)
        self.pos_embed = PositionalEncoding(dim=d_model, dropout=dropout, max_len=max_caption_len+4)

        # Feature fusion module
        #self.r2l_feat_fusion = FFNFeatureFusion(d_model=d_model, num_features=3, dropout=dropout)
        #self.l2r_feat_fusion = FFNFeatureFusion(d_model=d_model, num_features=3, dropout=dropout)
        
        # self.encoder_big = Encoder(n_layers, EncoderLayer(d_model, c(attn_big), c(feed_forward), dropout), d_model)
        #self.r2l_img_encoder_big = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads_big, num_layers=n_enc_layers, dropout=dropout, use_rope=False)
        #self.r2l_mot_encoder_big = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads_big, num_layers=n_enc_layers, dropout=dropout, use_rope=False)
        #self.r2l_obj_encoder_big = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads_big, num_layers=n_enc_layers, dropout=dropout, use_rope=False)
        self.r2l_encoder_big = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads_big, num_layers=n_enc_layers, dropout=dropout, use_rope=False)

        # self.encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout), d_model)
        #self.l2r_img_encoder = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_enc_layers, dropout=dropout, use_rope=False)
        #self.l2r_mot_encoder = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_enc_layers, dropout=dropout, use_rope=False)
        #self.l2r_obj_encoder = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_enc_layers, dropout=dropout, use_rope=False)
        self.l2r_encoder = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_enc_layers, dropout=dropout, use_rope=False)

        self.r2l_decoder = R2L_Decoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_dec_layers, dropout=dropout, use_rope=False)
        self.l2r_decoder = L2R_Decoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_dec_layers, dropout=dropout, use_rope=False)

        # self.generator = Generator(d_model, vocab.n_vocabs)
        self.r2l_generator = Generator(d_model=d_model, vocab_size=vocab.n_vocabs)
        self.l2r_generator = Generator(d_model=d_model, vocab_size=vocab.n_vocabs)

    def encode(self, src, src_mask, feature_mode_two=False):
        # ============== Spatial-Temporal Encoding ==============
        if feature_mode_two:
            x1 = self.r2l_image_src_embed(src[0])  # shape (64, 10, 512)
            x2 = self.r2l_motion_src_embed(src[1]) # shape (64, 10, 512)
            x3 = self.r2l_object_src_embed(src[2]) # shape (64, 10, 512)

            B, _, D = x1.shape
            _stacked = torch.stack([x1, x2, x3], dim=2)
            r2l_x = _stacked.reshape(B, -1, D)
            assert torch.equal(r2l_x[:,0::3,:], x1[:,:,:]) == True
            assert torch.equal(r2l_x[:,1::3,:], x2[:,:,:]) == True
            assert torch.equal(r2l_x[:,2::3,:], x3[:,:,:]) == True

            r2l_x = self.pos_embed(r2l_x)
            return self.r2l_encoder_big(r2l_x, src_mask)
            #return torch.cat((x1, x2, x3), dim=1)
            #assert x1.shape == x2.shape == x3.shape, f"x1.shape[{x1.shape}], x2.shape[{x2.shape}], x3.shape[{x3.shape}]"
            #print(f"x1.shape: {x1.shape}")
            #return self.r2l_feat_fusion([x1, x2, x3])

        # ============== Object-Relation Encoding ==============
        else:
            x1 = self.l2r_image_src_embed(src[0])
            x2 = self.l2r_motion_src_embed(src[1])
            x3 = self.l2r_object_src_embed(src[2])

            B, _, D = x1.shape
            _stacked = torch.stack([x1, x2, x3], dim=2)
            l2r_x = _stacked.reshape(B, -1, D)
            assert torch.equal(l2r_x[:,0::3,:], x1[:,:,:]) == True
            assert torch.equal(l2r_x[:,1::3,:], x2[:,:,:]) == True
            assert torch.equal(l2r_x[:,2::3,:], x3[:,:,:]) == True

            l2r_x = self.pos_embed(l2r_x)
            return self.l2r_encoder(l2r_x, src_mask)
            #return torch.cat((x1, x2, x3), dim=1)
            #assert x1.shape == x2.shape == x3.shape, f"x1.shape[{x1.shape}], x2.shape[{x2.shape}], x3.shape[{x3.shape}]"
            #print(f"x1.shape: {x1.shape}")
            # return self.feat_fusion([x1, x2, x3, x4])
            #return self.l2r_feat_fusion([x1, x2, x3])

    def r2l_decode(self, r2l_trg, memory, src_mask, r2l_trg_mask):
        x = self.r2l_trg_embed(r2l_trg)
        #print(f"[r2l_decode] emb.shape: {x.shape}")
        #print(f"[r2l_decode] memory.shape: {memory.shape}")
        x = self.pos_embed(x)
        return self.r2l_decoder(x, memory, src_mask, r2l_trg_mask)

    def l2r_decode(self, trg, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        x = self.l2r_trg_embed(trg)
        #print(f"[l2r_decode] emb.shape: {x.shape}")
        #print(f"[l2r_decode] memory.shape: {memory.shape}")
        #print(f"[l2r_decode] r2l_memory.shape: {r2l_memory.shape}")
        x = self.pos_embed(x)
        return self.l2r_decoder(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)

    def forward(self, src, r2l_trg, trg, mask):
        src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask = mask
        
        enc_src_mask, dec_src_mask = src_mask
        r2l_encoding_outputs = self.encode(src, enc_src_mask, feature_mode_two=True)
        #print("Finish r2l-enc")
        encoding_outputs = self.encode(src, enc_src_mask)
        #print("Finish l2r-enc")
        #print(f"r2l_enc_out.shape: {r2l_encoding_outputs.shape}")
        #print(f"l2r_enc_out.shape: {encoding_outputs.shape}")
        #print(f"r2l_trg.shape: {r2l_trg.shape}")
        #print(f"trg.shape: {trg.shape}")

        r2l_outputs = self.r2l_decode(r2l_trg, r2l_encoding_outputs, dec_src_mask, r2l_trg_mask)
        #print("Finish r2l-dec")
        l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, r2l_outputs, r2l_pad_mask)
        #print("Finish l2r-dec")
        #print(f"r2l_outputs.shape: {r2l_outputs.shape}")
        #print(f"l2r_outputs.shape: {l2r_outputs.shape}")

        # r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, dec_src_mask, r2l_trg_mask)
        # l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, None, None)


        # r2l_pred = self.generator(r2l_outputs)
        # l2r_pred = self.generator(l2r_outputs)
        r2l_pred = self.r2l_generator(r2l_outputs)
        l2r_pred = self.l2r_generator(l2r_outputs)
        #print(f"r2l_pred.shape: {r2l_pred.shape}")
        #print(f"l2r_pred.shape: {l2r_pred.shape}")

        #raise NotImplementedError("You should implement this function.")
        
        return r2l_pred, l2r_pred


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
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long, device=self.device)) 
                      for _ in range(batch_size)]
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
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
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
        hypotheses = [copy.deepcopy(torch.full((1, 1), start_symbol, dtype=torch.long, device=self.device)) 
                      for _ in range(batch_size)]
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
                src_mask_l += [dec_src_mask[i:i + 1]] * cur_beam_size
                r2l_memory_l += [r2l_memory[i: i + 1]] * cur_beam_size
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            r2l_memory_cur = torch.cat(r2l_memory_l, dim=0)
            "shape (sum(4 bt * cur_beam_sz_i), 1 dec_sent_len, 128 d_model)"
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
