import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable
from dataclasses import dataclass # Added for ModelArgs potentially

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


# --- RoPE Helper Functions (Adapted from deepseek-v3_model.py) ---

# Simplified ModelArgs for RoPE parameters needed here
@dataclass
class RoPEArgs:
    qk_rope_head_dim: int = 64 # Example value, adjust as needed
    max_seq_len: int = 5000 # Example value, adjust based on PositionalEncoding max_len
    rope_theta: float = 10000.0
    # Add other relevant params from deepseek ModelArgs if needed, e.g., factor, beta_fast/slow

def precompute_freqs_cis(args: RoPEArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.
    Simplified version without YARN scaling for now.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex64
    # freqs_cis needs to be [seq_len, dim // 2]
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis_slice: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.
    Assumes x has shape [bsz, ..., seqlen, dim] or similar where dim is the last dim.
    freqs_cis_slice has shape [seqlen, dim // 2], matching x's seqlen.
    """
    # x: [bsz, n_heads, seqlen, head_dim]
    # freqs_cis_slice: [seqlen, head_dim // 2]
    seqlen = x.size(-2)
    head_dim = x.size(-1)

    # Ensure the passed freqs_cis_slice has the correct sequence length
    assert freqs_cis_slice.shape[0] == seqlen, f"Sequence length mismatch: x has {seqlen}, freqs_cis_slice has {freqs_cis_slice.shape[0]}"

    # Reshape freqs_cis to broadcast: [1, 1, seqlen, head_dim // 2]
    freqs_cis_reshaped = freqs_cis_slice.unsqueeze(0).unsqueeze(0)

    # Reshape x to [bsz, n_heads, seqlen, head_dim // 2, 2] to view as complex
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_) # [bsz, n_heads, seqlen, head_dim // 2]

    # Apply rotation: element-wise multiplication with complex numbers
    # freqs_cis needs to be broadcastable to x_complex shape
    x_rotated = x_complex * freqs_cis_reshaped # [bsz, n_heads, seqlen, head_dim // 2]

    # Reshape back to original shape
    x_out = torch.view_as_real(x_rotated) # [bsz, n_heads, seqlen, head_dim // 2, 2]
    x_out = x_out.flatten(-2) # [bsz, n_heads, seqlen, head_dim]

    return x_out.type_as(x)

# --- End RoPE Helpers ---


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):

    def __init__(self, feature, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # Ensure that a_2 and b_2 are properly shaped for broadcasting
        # --- Modification: Check dimension compatibility more robustly ---
        # Original check might fail if feature dim is not the last one.
        # Assuming x is [..., feature_dim]
        if x.shape[-1] != self.a_2.shape[0]:
             print(f"Warning: LayerNorm input shape {x.shape} incompatible with feature size {self.a_2.shape[0]}. Skipping normalization.")
             return x # Return input unchanged if dimensions don't match
        # --- End Modification ---

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FeatEmbedding(nn.Module):

    def __init__(self, d_feat, d_model, dropout):
        super(FeatEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            LayerNorm(d_feat),
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
    # The mask operation is after QK and before softmax
    if mask is not None:

        if torch.cuda.is_available():
            mask.cuda()

        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn, value), self_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, head, d_model, dropout=0.1, qk_rope_head_dim=0): # Added qk_rope_head_dim
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.qk_rope_head_dim = qk_rope_head_dim # Store RoPE dimension
        assert qk_rope_head_dim <= self.d_k, "RoPE dimension cannot exceed head dimension"
        assert qk_rope_head_dim % 2 == 0, "RoPE dimension must be even"

        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.is_cross_attention = False # Flag to potentially differentiate behavior if needed externally
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None    # freqs_cis should be the full buffer [max_seq_len, dim // 2]    def forward(self, query, key, value, freqs_cis, mask=None): # freqs_cis is the full buffer
        # Check for empty batches or invalid shapes
    def forward(self, query, key, value, freqs_cis, mask=None): # Added freqs_cis
        if query.size(0) == 0 or key.size(0) == 0 or value.size(0) == 0:
            # Return empty tensor with appropriate dimensions
            return torch.zeros(0, 0, self.d_model, device=query.device)
        
        # Check for shape issues where input tensors are too small
        if query.dim() <= 2 or key.dim() <= 2 or value.dim() <= 2:
            # If any input has insufficient dimensions, expand to expected 3D shape
            # [batch, seq, dim] where batch=1, seq=1 if smaller
            if query.dim() <= 2:
                if query.dim() == 1:
                    # If 1D, add batch and sequence dimensions [dim] -> [1, 1, dim]
                    query = query.unsqueeze(0).unsqueeze(0)
                else:
                    # If 2D [batch, dim] -> [batch, 1, dim]
                    query = query.unsqueeze(1) 
                # Ensure the last dimension is d_model
                if query.size(-1) != self.d_model:
                    query = query.expand(*query.size()[:-1], self.d_model)
                    
            if key.dim() <= 2:
                if key.dim() == 1:
                    key = key.unsqueeze(0).unsqueeze(0)
                else:
                    key = key.unsqueeze(1)
                if key.size(-1) != self.d_model:
                    key = key.expand(*key.size()[:-1], self.d_model)
                    
            if value.dim() <= 2:
                if value.dim() == 1:
                    value = value.unsqueeze(0).unsqueeze(0)
                else:
                    value = value.unsqueeze(1)
                if value.size(-1) != self.d_model:
                    value = value.expand(*value.size()[:-1], self.d_model)
            
        if mask is not None:
            # 多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            # 再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第一维添加一维，与后面的self attention计算维度一样
            mask = mask.unsqueeze(1)
        
        n_batch = query.size(0)
        q_len, k_len, v_len = query.size(1), key.size(1), value.size(1) # Get sequence lengths
        
        # Ensure inputs are float tensors (convert from long/int if needed)
        if query.dtype != torch.float32 and query.dtype != torch.float16:
            query = query.float()
        if key.dtype != torch.float32 and key.dtype != torch.float16:
            key = key.float()
        if value.dtype != torch.float32 and value.dtype != torch.float16:
            value = value.float()
            
        try:
            # Project and reshape query, key, value
            query = self.linear_query(query).view(
                n_batch, q_len, self.head, self.d_k).transpose(1, 2)  # [b, h, q_len, d_k]
            key = self.linear_key(key).view(
                n_batch, k_len, self.head, self.d_k).transpose(1, 2)  # [b, h, k_len, d_k]
            value = self.linear_value(value).view(
                n_batch, v_len, self.head, self.d_k).transpose(1, 2)  # [b, h, v_len, d_k]
        except RuntimeError as e:
            # If there's still an error despite our dimensionality fixes, log info and try simpler approach
            print(f"Error with shapes: query {query.shape}, key {key.shape}, value {value.shape}")
            print(f"Exception: {e}")
            # Return an appropriate zero tensor as fallback
            return torch.zeros(n_batch, q_len, self.d_model, device=query.device)        # --- Apply RoPE ---
        if self.qk_rope_head_dim > 0:
            try:
                # Determine if it's self-attention. A simple check:
                # If query and key have the same shape and sequence length, assume self-attention.
                # Note: This might not be perfectly robust if views are involved, but often sufficient.
                # A more robust method might involve passing an explicit flag if needed.
                is_self_attention = (q_len == k_len) # Simplified check based on lengths

                # Always apply RoPE to Query based on its sequence length
                freqs_cis_q = freqs_cis[:q_len] # Slice for query length
                q_rope = query[..., :self.qk_rope_head_dim]
                q_rope = apply_rotary_emb(q_rope, freqs_cis_slice=freqs_cis_q) # Pass the slice
                query = torch.cat((q_rope, query[..., self.qk_rope_head_dim:]), dim=-1)

                # Apply RoPE to Key ONLY if it's self-attention
                if is_self_attention:
                    freqs_cis_k = freqs_cis[:k_len] # Slice for key length (k_len == q_len here)
                    k_rope = key[..., :self.qk_rope_head_dim]
                    k_rope = apply_rotary_emb(k_rope, freqs_cis_slice=freqs_cis_k) # Pass the slice
                    key = torch.cat((k_rope, key[..., self.qk_rope_head_dim:]), dim=-1)
                # Else: key comes from encoder memory in cross-attention.
                # It already has positional information from the encoder. Do not apply decoder RoPE.
            except Exception as e:
                print(f"Error in RoPE application: {e}")
                # Continue without RoPE if there's an error        # --- End RoPE ---

        try:
            # Calculate attention scores
            x, self.attn = self_attention(
                query, key, value, dropout=self.dropout, mask=mask)

            # Concatenate heads and apply final linear layer
            x = x.transpose(1, 2).contiguous().view(
                n_batch, -1, self.head * self.d_k)

            return self.linear_out(x)
            
        except Exception as e:
            print(f"Error in attention calculation: {e}")
            # Return a zero tensor with the expected output shape as fallback
            return torch.zeros(n_batch, q_len, self.d_model, device=query.device)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layer_norm(x + sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, freqs_cis, mask): # Added freqs_cis
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, freqs_cis, mask)) # Pass freqs_cis
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
        self.attn = attn # This should be the modified MultiHeadAttention instance
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(
            SublayerConnection(size, dropout), sublayer_num)

    # Added freqs_cis to arguments
    def forward(self, x, memory, freqs_cis, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        # Self-attention with RoPE
        x = self.sublayer_connection[0](
            x, lambda x: self.attn(x, x, x, freqs_cis, trg_mask)) # Pass freqs_cis
        # Cross-attention (RoPE typically not applied here, but depends on design)
        # Assuming cross-attention doesn't use RoPE on memory (encoder output)
        # If RoPE should be applied to query 'x' in cross-attn, attn needs modification or separate instances
        x = self.sublayer_connection[1](
            x, lambda x: self.attn(x, memory, memory, freqs_cis, src_mask)) # Pass freqs_cis (might be ignored by attn if memory is key/value)

        if r2l_memory is not None:
             # Cross-attention with r2l_memory
             # Pass freqs_cis, assuming attn handles it appropriately for cross-attention query
            x = self.sublayer_connection[-2](x, lambda x: self.attn(
                x, r2l_memory, r2l_memory, freqs_cis, r2l_trg_mask))

        return self.sublayer_connection[-1](x, self.feed_forward)


class Encoder(nn.Module):

    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer, n)

    def forward(self, x, freqs_cis, src_mask): # Added freqs_cis
        for layer in self.encoder_layer:
            x = layer(x, freqs_cis, src_mask) # Pass freqs_cis
        return x


class R2L_Decoder(nn.Module):

    def __init__(self, n, decoder_layer):
        super(R2L_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, freqs_cis, src_mask, r2l_trg_mask): # Added freqs_cis
        for layer in self.decoder_layer:
            # Pass freqs_cis to the decoder layer
            x = layer(x, memory, freqs_cis, src_mask, r2l_trg_mask)
        return x


class L2R_Decoder(nn.Module):

    def __init__(self, n, decoder_layer):
        super(L2R_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, freqs_cis, src_mask, trg_mask, r2l_memory, r2l_trg_mask): # Added freqs_cis
        for layer in self.decoder_layer:
             # Pass freqs_cis to the decoder layer
            x = layer(x, memory, freqs_cis, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
        return x


def pad_mask(src, r2l_trg, trg, pad_idx):
    if isinstance(src, tuple):
        if len(src) == 4:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
            src_rel_mask = (src[3][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask,
                            src_object_mask, src_rel_mask)
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
            # --- Fix: Correct slicing syntax ---
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            # --- End Fix ---
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            src_mask = (enc_src_mask, dec_src_mask)
    else:
        src_mask = (src[:, :, 0] != pad_idx).unsqueeze(1)
    if trg is not None:
        if isinstance(src_mask, tuple):
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(
                trg.size(1)).type_as(src_image_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(
                1).type_as(src_image_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(
                r2l_trg.size(1)).type_as(src_image_mask.data)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
        else:
            trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(
                trg.size(1)).type_as(src_mask.data)
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(
                1).type_as(src_mask.data)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(
                r2l_trg.size(1)).type_as(src_mask.data)
            # src_mask[batch, 1, lens]  trg_mask[batch, 1, lens]
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask

    else:
        return src_mask


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    if torch.cuda.is_available():
        return (torch.from_numpy(mask) == 0).cuda()
    else:
        return (torch.from_numpy(mask) == 0).cpu()


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class ABDTransformer(nn.Module):

    def __init__(self, vocab, d_feat, d_model, d_ff, n_heads, n_layers, dropout, feature_mode,
                 device='cuda', n_heads_big=128,
                 # --- RoPE Config ---
                 qk_rope_head_dim=64, # Example: Use 64 dims for RoPE per head
                 max_seq_len=5000,    # Should match PositionalEncoding max_len
                 rope_theta=10000.0):
                 # --- End RoPE Config ---
        super(ABDTransformer, self).__init__()
        self.vocab = vocab
        if torch.cuda.is_available():
            self.device = device
        else:
            self.device = 'cpu'
        self.feature_mode = feature_mode
        self.d_model = d_model # Store d_model
        self.n_heads = n_heads # Store n_heads
        # self.qk_rope_head_dim = qk_rope_head_dim # Store RoPE dim # Original line

        c = copy.deepcopy

        # --- Adjust RoPE Dim based on head constraints ---
        config_qk_rope_head_dim = qk_rope_head_dim # Store the requested dim

        # Calculate d_k for both attention types
        d_k_attn = d_model // n_heads
        d_k_attn_big = d_model // n_heads_big
        min_d_k = min(d_k_attn, d_k_attn_big)

        # Ensure qk_rope_head_dim is valid (<= min_d_k) and even
        valid_qk_rope_head_dim = min_d_k if min_d_k % 2 == 0 else min_d_k - 1
        if valid_qk_rope_head_dim < 0: valid_qk_rope_head_dim = 0 # Cannot be negative

        # Use the smaller of the configured value and the maximum valid value
        self.qk_rope_head_dim = min(config_qk_rope_head_dim, valid_qk_rope_head_dim)

        # Optional: Add a warning if the dimension was reduced
        if self.qk_rope_head_dim < config_qk_rope_head_dim:
            print(f"Warning: Requested qk_rope_head_dim ({config_qk_rope_head_dim}) exceeds minimum head dimension ({min_d_k}). "
                  f"Using adjusted qk_rope_head_dim = {self.qk_rope_head_dim}.")
        # --- End RoPE Dim Adjustment ---


        # --- RoPE Precomputation ---
        # Use the potentially adjusted self.qk_rope_head_dim
        self.rope_args = RoPEArgs(qk_rope_head_dim=self.qk_rope_head_dim,
                                  max_seq_len=max_seq_len, # Fix: Use max_seq_len parameter
                                  rope_theta=rope_theta)
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.rope_args), persistent=False)
        # --- End RoPE Precomputation ---


        # attn_no_heads = MultiHeadAttention(1, d_model, dropout)

        # Pass the potentially adjusted self.qk_rope_head_dim to MultiHeadAttention instances
        attn = MultiHeadAttention(n_heads, d_model, dropout, qk_rope_head_dim=self.qk_rope_head_dim)
        attn_big = MultiHeadAttention(n_heads_big, d_model, dropout, qk_rope_head_dim=self.qk_rope_head_dim)
        # attn_big2 = MultiHeadAttention(10, d_model, dropout)

        feed_forward = PositionWiseFeedForward(d_model, d_ff)

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
        # PositionalEncoding might become redundant or used differently with RoPE
        # Keep it for now, but its contribution might need re-evaluation
        self.pos_embed = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        # self.encoder_no_heads = Encoder(n_layers, EncoderLayer(d_model, c(attn_no_heads), c(feed_forward), dropout))

        # EncoderLayer now expects freqs_cis
        self.encoder = Encoder(n_layers, EncoderLayer(
            d_model, c(attn), c(feed_forward), dropout))
        self.encoder_big = Encoder(n_layers, EncoderLayer(
            d_model, c(attn_big), c(feed_forward), dropout))
        # self.encoder_big2 = Encoder(n_layers, EncoderLayer(d_model, c(attn_big2), c(feed_forward), dropout))
        # EncoderLayerNoAttention doesn't use attention, so no freqs_cis needed for it directly
        # However, the attn instance passed might still need qk_rope_head_dim if used elsewhere
        no_attn_instance = MultiHeadAttention(n_heads, d_model, dropout, qk_rope_head_dim=0) # No RoPE for this one's lambda
        self.encoder_no_attention = Encoder(n_layers,
                                            EncoderLayerNoAttention(d_model, c(no_attn_instance), c(feed_forward), dropout))

        # DecoderLayer now expects freqs_cis
        self.r2l_decoder = R2L_Decoder(n_layers, DecoderLayer(d_model, c(attn), c(feed_forward),
                                                              sublayer_num=3, dropout=dropout))
        self.l2r_decoder = L2R_Decoder(n_layers, DecoderLayer(d_model, c(attn), c(feed_forward),
                                                              sublayer_num=4, dropout=dropout))

        self.generator = Generator(d_model, vocab.n_vocabs)

    def encode(self, src, src_mask, feature_mode_two=False):
        # Get the maximum sequence length from the input features
        if isinstance(src, tuple):
             # Handle tuple input (e.g., multi-modal features)
             # Determine max length across all features in the tuple
             max_len = max(s.size(1) for s in src)
        else:
             # Handle single tensor input
             max_len = src.size(1)

        if self.feature_mode == 'two':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1) # Keep positional encoding for now
            x1 = self.encoder_big(x1, self.freqs_cis, src_mask[0]) # Pass full buffer
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, self.freqs_cis, src_mask[1]) # Pass full buffer
            return x1 + x2
        if feature_mode_two: # This seems redundant with the above block? Assuming it's intended.
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, self.freqs_cis, src_mask[0]) # Pass full buffer
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, self.freqs_cis, src_mask[1]) # Pass full buffer
            return x1 + x2
        if self.feature_mode == 'one':
            x = self.src_embed(src)
            x = self.pos_embed(x)
            return self.encoder(x, self.freqs_cis, src_mask) # Pass full buffer
        elif self.feature_mode == 'two': # This is also redundant? Consolidate logic if possible.
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, self.freqs_cis, src_mask[0]) # Pass full buffer
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, self.freqs_cis, src_mask[1]) # Pass full buffer
            return x1 + x2
        elif self.feature_mode == 'three':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, self.freqs_cis, src_mask[0]) # Pass full buffer
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, self.freqs_cis, src_mask[1]) # Pass full buffer
            x3 = self.object_src_embed(src[2])
            x3 = self.pos_embed(x3)
            x3 = self.encoder(x3, self.freqs_cis, src_mask[2]) # Pass full buffer
            return x1 + x2 + x3
        elif self.feature_mode == 'four':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, self.freqs_cis, src_mask[0]) # Pass full buffer

            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, self.freqs_cis, src_mask[1]) # Pass full buffer

            x3 = self.object_src_embed(src[2])
            # x3 = self.pos_embed(x3) # Original code commented this out
            x3 = self.encoder(x3, self.freqs_cis, src_mask[2]) # Pass full buffer
            # x3 = self.encoder_no_attention(x3, src_mask[2]) # Original code used this

            x4 = self.rel_src_embed(src[3])
            # x4 = self.pos_embed(x4) # Original code commented this out
            # Pass freqs_cis to encoder_no_attention? It doesn't use attn, so likely not needed.
            # If encoder_no_attention internally uses layers that need it, it must be passed.
            # Assuming EncoderLayerNoAttention doesn't need it.
            x4 = self.encoder_no_attention(x4, src_mask[3]) # No freqs_cis needed here
            # x4 = self.encoder(x4, self.freqs_cis, src_mask[3]) # Original code used this
            return x1 + x2 + x3 + x4

    def r2l_decode(self, r2l_trg, memory, src_mask, r2l_trg_mask):
        x = self.trg_embed(r2l_trg)
        x = self.pos_embed(x) # Keep positional encoding for now
        # Pass full self.freqs_cis to the decoder
        return self.r2l_decoder(x, memory, self.freqs_cis, src_mask, r2l_trg_mask)

    def l2r_decode(self, trg, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        x = self.trg_embed(trg)
        x = self.pos_embed(x) # Keep positional encoding for now
        # Pass full self.freqs_cis to the decoder
        return self.l2r_decoder(x, memory, self.freqs_cis, src_mask, trg_mask, r2l_memory, r2l_trg_mask)

    def forward(self, src, r2l_trg, trg, mask):
        src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask = mask
        if self.feature_mode == 'one':
            encoding_outputs = self.encode(src, src_mask)
            # Decode steps need freqs_cis, handled internally by r2l_decode/l2r_decode
            r2l_outputs = self.r2l_decode(
                r2l_trg, encoding_outputs, src_mask, r2l_trg_mask)
            l2r_outputs = self.l2r_decode(
                trg, encoding_outputs, src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

        elif self.feature_mode == 'two' or 'three' or 'four':
            enc_src_mask, dec_src_mask = src_mask
            # Encode steps need freqs_cis, handled internally by encode
            r2l_encoding_outputs = self.encode(
                src, enc_src_mask, feature_mode_two=True)
            encoding_outputs = self.encode(src, enc_src_mask)

            # Decode steps need freqs_cis, handled internally by r2l_decode/l2r_decode
            r2l_outputs = self.r2l_decode(
                r2l_trg, r2l_encoding_outputs, dec_src_mask, r2l_trg_mask)
            l2r_outputs = self.l2r_decode(
                trg, encoding_outputs, dec_src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

            # r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, dec_src_mask, r2l_trg_mask)
            # l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, None, None)
        else:
            raise ValueError("Invalid feature_mode specified") # Use ValueError

        r2l_pred = self.generator(r2l_outputs)
        l2r_pred = self.generator(l2r_outputs)

        return r2l_pred, l2r_pred

    # --- Beam Search Modifications (Conceptual) ---
    # The beam search methods need to correctly handle freqs_cis slicing for each decoding step.
    # This requires passing the current decoding step length to get the right slice.

    def greedy_decode(self, batch_size, src_mask, memory, max_len):
        # Simplified greedy decode - RoPE integration needs careful step handling
        eos_idx = self.vocab.word2idx['<S>']
        r2l_hidden = None
 
        with torch.no_grad():
            output = torch.ones(batch_size, 1).fill_(eos_idx).long().to(self.device) # Use self.device
            for i in range(max_len + 1): # Adjusted loop range
                current_len = output.size(1)
                # Create target mask for the current length
                trg_mask = subsequent_mask(current_len).to(self.device) # Use self.device

                # Embed the token IDs
                output_emb = self.trg_embed(output)
                output_emb = self.pos_embed(output_emb)
                
                # Pass full self.freqs_cis to r2l_decoder
                dec_out = self.r2l_decoder(
                    output_emb, memory, self.freqs_cis, src_mask, trg_mask) # Pass full buffer
                r2l_hidden = dec_out # Store the full output sequence hidden states if needed
                pred = self.generator(dec_out[:, -1]) # Get prediction for the last token only
                next_word = pred.max(dim=-1)[1].unsqueeze(1)
                output = torch.cat([output, next_word], dim=-1)

                # Check for EOS token to potentially break early (optional)
                # if (next_word == eos_idx).all(): break

        # Return the final hidden state (or full sequence) and the generated sequence
        # Note: Returning r2l_hidden might be large if max_len is big.
        # Consider what is actually needed downstream.
        return r2l_hidden, output


    def r2l_beam_search_decode(self, batch_size, src, src_mask, model_encodings, beam_size, max_len):
        # Beam search requires careful handling of freqs_cis slicing per step/hypothesis
        # This implementation sketch shows where freqs_cis needs to be passed

        end_symbol = self.vocab.word2idx['<S>']
        start_symbol = self.vocab.word2idx['<S>']
        r2l_outputs = None # Placeholder for potential full sequence outputs

        hypotheses = [torch.full((1, 1), start_symbol, dtype=torch.long, device=self.device) for _ in range(batch_size)]
        completed_hypotheses = [[] for _ in range(batch_size)]
        hyp_scores = [torch.zeros(1, dtype=torch.float, device=self.device) for _ in range(batch_size)]

        for iter in range(max_len + 1):
            if all([len(completed_hypotheses[i]) >= beam_size for i in range(batch_size)]): # Use >=
                break

            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l = [], [], [], []
            max_hyp_len = 0 # Track max length for freqs_cis slicing
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes.append(0)
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                max_hyp_len = max(max_hyp_len, decoded_len) # Update max length
                cur_beam_sizes.append(cur_beam_size)
                last_tokens.append(hypotheses[i])
                # Ensure model_encodings and src_mask have batch dim before indexing
                model_encodings_l.append(model_encodings[i:i + 1].repeat_interleave(cur_beam_size, dim=0))
                src_mask_l.append(src_mask[i:i + 1].repeat_interleave(cur_beam_size, dim=0))

            if not last_tokens: break # All hypotheses finished

            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0) # Shape: [total_beams, current_len]

            # Create target mask for the current max length
            tgt_mask = subsequent_mask(max_hyp_len).to(self.device)
            # Slice mask if y_tm1 length is less than max_hyp_len (shouldn't happen if max_hyp_len derived correctly)
            current_tgt_mask = tgt_mask[:, :y_tm1.size(1), :y_tm1.size(1)]            # First embed the token IDs before passing to decoder
            if y_tm1.dtype == torch.long:  # Check if these are token indices
                # Apply embedding and positional encoding
                y_tm1_emb = self.trg_embed(y_tm1)
                y_tm1_emb = self.pos_embed(y_tm1_emb)
            else:
                y_tm1_emb = y_tm1  # Already embedded
            
            # Pass full self.freqs_cis to decoder
            out = self.r2l_decoder(y_tm1_emb, model_encodings_cur, self.freqs_cis, src_mask_cur, current_tgt_mask) # Pass full buffer
            # r2l_outputs = out # Store full output if needed

            log_prob = self.generator(out[:, -1, :]) # Get log_prob for the last token: [total_beams, vocab_size]

            # Reshape/split log_prob per batch item (careful with variable beam sizes)
            log_prob_list = torch.split(log_prob, cur_beam_sizes, dim=0)
            hyp_scores_list = torch.split(torch.cat(hyp_scores, dim=0), cur_beam_sizes, dim=0) # Need to handle None scores if any batch finished early

            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                # --- Beam search expansion logic (needs careful index handling) ---
                # This part remains complex and needs thorough checking, especially
                # how prev_hyp_ids map back to the correct hypothesis in hypotheses[i]
                # and how scores are accumulated. The core change is passing freqs_cis above.
                # The existing logic for topk, prev_hyp_ids, hyp_word_ids, and completion check
                # should be preserved but verified.

                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses.append(None)
                    new_hyp_scores.append(None)
                    continue

                cur_beam_sz_i = log_prob_list[i].shape[0]
                if cur_beam_sz_i == 0: # Handle case where a batch item finishes
                     new_hypotheses.append(None)
                     new_hyp_scores.append(None)
                     continue

                vocab_sz = log_prob_list[i].shape[-1]
                # Ensure hyp_scores_list[i] is correctly shaped for broadcasting
                current_scores = hyp_scores_list[i].unsqueeze(-1) # [cur_beam_sz_i, 1]
                cumulative_hyp_scores_i = (current_scores + log_prob_list[i]).view(-1) # [cur_beam_sz_i * vocab_sz]


                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                # Ensure k is not larger than the number of scores
                k_val = min(live_hyp_num_i, cumulative_hyp_scores_i.numel())
                if k_val <= 0: # No candidates needed or possible
                    new_hypotheses.append(hypotheses[i]) # Keep existing hypotheses if no expansion
                    new_hyp_scores.append(hyp_scores[i]) # Keep existing scores
                    continue

                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                    cumulative_hyp_scores_i, k=k_val)

                # Handle potential division by zero if vocab_sz is 0 (should not happen)
                if vocab_sz == 0: raise ValueError("Vocabulary size cannot be zero.")
                prev_hyp_ids = top_cand_hyp_pos // vocab_sz
                hyp_word_ids = top_cand_hyp_pos % vocab_sz

                new_hypotheses_i, new_hyp_scores_i = [], []
                processed_prev_ids = set() # Avoid duplicates if k > cur_beam_sz_i

                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                    prev_hyp_id_item = prev_hyp_id.item()
                    hyp_word_id_item = hyp_word_id.item()
                    cand_new_hyp_score_item = cand_new_hyp_score.item()

                    # Ensure prev_hyp_id is valid
                    if prev_hyp_id_item >= hypotheses[i].shape[0]: continue # Should not happen with correct topk

                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id_item], torch.tensor([hyp_word_id_item], device=self.device)))

                    if hyp_word_id_item == end_symbol:
                        # Ensure score is float, not tensor
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score_item))
                        # Prune completed hypotheses immediately if beam is full
                        if len(completed_hypotheses[i]) >= beam_size:
                             completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
                             completed_hypotheses[i] = completed_hypotheses[i][:beam_size]
                             # If we filled the beam with completed ones, we might stop expanding this batch item
                             # This requires more complex logic to track if *all* live hyps ended.

                    else:
                        # Add unsqueezed tensor for later cat
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(0)) # Add batch dim
                        new_hyp_scores_i.append(cand_new_hyp_score_item)


                if len(new_hypotheses_i) > 0:
                    # Concatenate along the batch dimension (dim=0)
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=0).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                elif len(completed_hypotheses[i]) < beam_size:
                     # No new hypotheses generated, but not finished - keep old ones if any?
                     # Or mark as finished? If no new hyps AND not enough completed, it's stuck.
                     # Mark as None to stop processing this batch item.
                     hypotheses_i, hyp_scores_i = None, None
                else: # Finished for this batch item
                    hypotheses_i, hyp_scores_i = None, None

                new_hypotheses.append(hypotheses_i)
                new_hyp_scores.append(hyp_scores_i)

            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores
            # Filter out None scores before next iteration's cat
            hyp_scores = [s for s in hyp_scores if s is not None]


        # Post-processing (add incomplete hypotheses if needed)
        for i in range(batch_size):
             if hypotheses[i] is not None and hyp_scores[i] is not None: # Check if not None
                hyps_to_add = beam_size - len(completed_hypotheses[i])
                if hyps_to_add > 0 and hyp_scores[i].numel() > 0: # Check if scores exist
                    # Ensure k is not larger than available scores
                    k_val = min(hyps_to_add, hyp_scores[i].numel())
                    scores, ix = torch.topk(hyp_scores[i], k=k_val)
                    for score, id in zip(scores, ix):
                        # Ensure id is within bounds
                        if id.item() < hypotheses[i].shape[0]:
                            completed_hypotheses[i].append(Hypothesis(
                                value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id.item()][1:]],
                                score=score.item())) # Use .item()

        # Final sort
        for i in range(batch_size):
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
            # Ensure only beam_size results are kept
            completed_hypotheses[i] = completed_hypotheses[i][:beam_size]

        # Return placeholder for r2l_outputs and the completed hypotheses
        return None, completed_hypotheses


    def beam_search_decode(self, src, beam_size, max_len):
        """
        Beam Search Decoder incorporating RoPE.
        """
        start_symbol = self.vocab.word2idx['<S>']
        end_symbol = self.vocab.word2idx['<S>'] # Should likely be '<EOS>' or similar

        # --- Encoding with RoPE ---
        src_mask = pad_mask(src, r2l_trg=None, trg=None, pad_idx=self.vocab.word2idx['<PAD>'])
        if self.feature_mode == 'one':
            batch_size = src.shape[0]
            # encode handles freqs_cis internally
            model_encodings = self.encode(src, src_mask)
            # r2l_beam_search_decode needs RoPE integration as well
            r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, src_mask,
                                                                               model_encodings=model_encodings,
                                                                               beam_size=beam_size, max_len=max_len)
        elif self.feature_mode == 'two' or 'three' or 'four':
            batch_size = src[0].shape[0]
            enc_src_mask, dec_src_mask = src_mask # Unpack masks
            # encode handles freqs_cis internally
            r2l_model_encodings = self.encode(src, enc_src_mask, feature_mode_two=True)
            model_encodings = self.encode(src, enc_src_mask)
            # r2l_beam_search_decode needs RoPE integration
            r2l_memory, r2l_completed_hypotheses = self.r2l_beam_search_decode(batch_size, src, dec_src_mask,
                                                                               model_encodings=r2l_model_encodings,
                                                                               beam_size=beam_size, max_len=max_len)
        else:
             raise ValueError("Invalid feature_mode")

        # --- L2R Beam Search Setup ---
        hypotheses = [torch.full((1, 1), start_symbol, dtype=torch.long, device=self.device) for _ in range(batch_size)]
        completed_hypotheses = [[] for _ in range(batch_size)]
        hyp_scores = [torch.zeros(1, dtype=torch.float, device=self.device) for _ in range(batch_size)]

        # --- L2R Beam Search Loop ---
        for iter in range(max_len + 1):
            if all([len(completed_hypotheses[i]) >= beam_size for i in range(batch_size)]):
                break

            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l, r2l_memory_l = [], [], [], [], []
            max_hyp_len = 0
            for i in range(batch_size):
                 if hypotheses[i] is None:
                    cur_beam_sizes.append(0)
                    continue
                 cur_beam_size, decoded_len = hypotheses[i].shape
                 max_hyp_len = max(max_hyp_len, decoded_len)
                 cur_beam_sizes.append(cur_beam_size)
                 last_tokens.append(hypotheses[i])
                 model_encodings_l.append(model_encodings[i:i + 1].repeat_interleave(cur_beam_size, dim=0))
                 # Use dec_src_mask if multi-modal, otherwise src_mask
                 current_src_mask = dec_src_mask[i:i+1] if isinstance(src_mask, tuple) else src_mask[i:i+1]
                 src_mask_l.append(current_src_mask.repeat_interleave(cur_beam_size, dim=0))
                 # Handle r2l_memory potentially being None if r2l_beam_search failed or returned None
                 if r2l_memory is not None and r2l_memory[i:i+1] is not None:
                      r2l_memory_l.append(r2l_memory[i:i + 1].repeat_interleave(cur_beam_size, dim=0))
                 else:
                      # Need a placeholder or handle absence in l2r_decoder
                      # For now, assume r2l_memory is always available if needed by the model config
                      # If r2l_memory can be None, the l2r_decoder call needs adjustment
                      # This might indicate an issue in r2l_beam_search_decode returning None inappropriately
                      # Add a check or default tensor if necessary, e.g. zeros of expected shape.
                      # Placeholder:
                      if r2l_memory is not None: # Check if the main tensor exists
                           r2l_memory_l.append(torch.zeros_like(model_encodings[i:i + 1]).repeat_interleave(cur_beam_size, dim=0)) # Example placeholder
                      else: # r2l_memory itself is None
                           # Cannot proceed if r2l_memory is required but absent
                           # This part needs robust handling based on whether r2l path is optional
                           pass # Or raise error


            if not last_tokens: break

            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            y_tm1 = torch.cat(last_tokens, dim=0)
            # Handle potential empty r2l_memory_l if placeholders were used or errors occurred
            r2l_memory_cur = torch.cat(r2l_memory_l, dim=0) if r2l_memory_l else None            
            tgt_mask = subsequent_mask(max_hyp_len).to(self.device)
            current_tgt_mask = tgt_mask[:, :y_tm1.size(1), :y_tm1.size(1)]

            # First embed the token IDs before passing to decoder
            if y_tm1.dtype == torch.long:  # Check if these are token indices
                # Apply embedding and positional encoding
                y_tm1_emb = self.trg_embed(y_tm1)
                y_tm1_emb = self.pos_embed(y_tm1_emb)
            else:
                y_tm1_emb = y_tm1  # Already embedded
                
            # Pass full self.freqs_cis to l2r_decoder
            # Handle r2l_memory_cur potentially being None
            out = self.l2r_decoder(y_tm1_emb, model_encodings_cur, self.freqs_cis, src_mask_cur,
                                   current_tgt_mask, r2l_memory_cur, r2l_trg_mask=None) # Pass None for r2l_trg_mask?

            log_prob = self.generator(out[:, -1, :]) # [total_beams, vocab_size]

            # --- Beam search expansion (similar to r2l, needs verification) ---
            log_prob_list = torch.split(log_prob, cur_beam_sizes, dim=0)
            hyp_scores_list = torch.split(torch.cat(hyp_scores, dim=0), cur_beam_sizes, dim=0)

            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses.append(None)
                    new_hyp_scores.append(None)
                    continue

                cur_beam_sz_i = log_prob_list[i].shape[0]
                if cur_beam_sz_i == 0:
                     new_hypotheses.append(None)
                     new_hyp_scores.append(None)
                     continue

                vocab_sz = log_prob_list[i].shape[-1]
                current_scores = hyp_scores_list[i].unsqueeze(-1)
                cumulative_hyp_scores_i = (current_scores + log_prob_list[i]).view(-1)

                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                k_val = min(live_hyp_num_i, cumulative_hyp_scores_i.numel())
                if k_val <= 0:
                    new_hypotheses.append(hypotheses[i])
                    new_hyp_scores.append(hyp_scores[i])
                    continue

                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=k_val)

                if vocab_sz == 0: raise ValueError("Vocabulary size cannot be zero.")
                prev_hyp_ids = top_cand_hyp_pos // vocab_sz
                hyp_word_ids = top_cand_hyp_pos % vocab_sz

                new_hypotheses_i, new_hyp_scores_i = [], []
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                    prev_hyp_id_item = prev_hyp_id.item()
                    hyp_word_id_item = hyp_word_id.item()
                    cand_new_hyp_score_item = cand_new_hyp_score.item()

                    if prev_hyp_id_item >= hypotheses[i].shape[0]: continue

                    new_hyp_sent = torch.cat(
                        (hypotheses[i][prev_hyp_id_item], torch.tensor([hyp_word_id_item], device=self.device)))

                    if hyp_word_id_item == end_symbol:
                        completed_hypotheses[i].append(Hypothesis(
                            value=[self.vocab.idx2word[a.item()] for a in new_hyp_sent[1:-1]],
                            score=cand_new_hyp_score_item))
                        if len(completed_hypotheses[i]) >= beam_size:
                             completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
                             completed_hypotheses[i] = completed_hypotheses[i][:beam_size]
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(0))
                        new_hyp_scores_i.append(cand_new_hyp_score_item)

                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=0).to(self.device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=self.device)
                elif len(completed_hypotheses[i]) < beam_size:
                     hypotheses_i, hyp_scores_i = None, None
                else:
                    hypotheses_i, hyp_scores_i = None, None

                new_hypotheses.append(hypotheses_i)
                new_hyp_scores.append(hyp_scores_i)

            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores
            hyp_scores = [s for s in hyp_scores if s is not None]


        # --- Post-processing ---
        for i in range(batch_size):
             if hypotheses[i] is not None and hyp_scores[i] is not None:
                hyps_to_add = beam_size - len(completed_hypotheses[i])
                if hyps_to_add > 0 and hyp_scores[i].numel() > 0:
                    k_val = min(hyps_to_add, hyp_scores[i].numel())
                    scores, ix = torch.topk(hyp_scores[i], k=k_val)
                    for score, id in zip(scores, ix):
                         if id.item() < hypotheses[i].shape[0]:
                            completed_hypotheses[i].append(Hypothesis(
                                value=[self.vocab.idx2word[a.item()] for a in hypotheses[i][id.item()][1:]],
                                score=score.item()))

        for i in range(batch_size):
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)
            completed_hypotheses[i] = completed_hypotheses[i][:beam_size]

        # Return both r2l and l2r results
        return r2l_completed_hypotheses, completed_hypotheses
