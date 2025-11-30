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


class FeatEmbedding(nn.Module):

    def __init__(self, d_feat, d_model, dropout):
        super(FeatEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            nn.LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_model)
        )

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
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_size = d_model // num_heads
        self.use_rope = use_rope
        self.attn_weights = None  # To store if needed

    def forward(self, query, key_value, mask=None):
        """
        query, key, value: [batch_size, seq_len, d_model]
        mask: Optional [batch_size, seq_q, seq_k] with 0/1 (0 to mask)
        """

        # Batch size, Sequence length, Embedding dimensionality
        B, T, C = query.shape

        # 1. Query Path
        q_final = self.W_q(query)

        # 2. Key-Value Path
        k_final = self.W_k(key_value)
        v_final = self.W_v(key_value)

        # TODO: 3. Apply Rotary Positional Embedding to Q, K
    
        # 4. Reshape Q, K, V for multi-head attention
        # (B, T, C) -> (B, T, num_heads, head_size) -> (B, num_heads, T, head_size)
        q_heads = q_final.view(B, -1, self.num_heads, self.head_size).transpose(1, 2)
        k_heads = k_final.view(B, -1, self.num_heads, self.head_size).transpose(1, 2)
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


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, multiple_of: int, dropout: float):
        super(FFN, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_ff),
            NewGELUActivation(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.transform(x)
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
        
        self.self_attn = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        if first_layer: self.self_attn_skipconn = SkipConnectionAfterLN(size=d_model, dropout=dropout)
        else:           self.self_attn_skipconn = SublayerConnection(size=d_model, dropout=dropout)
        
        self.cross_attn = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.cross_attn_skipconn = SublayerConnection(size=d_model, dropout=dropout)
        
        self.query_ffn = FFN(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, dropout=dropout)
        self.query_ffn_skipconn = SublayerConnection(size=d_model, dropout=dropout)

    def forward(self, query, feature, feat_mask):
        # --- Self-Attention ---
        query = self.self_attn_skipconn(query, lambda x: self.self_attn(x, x, None))
        # --- Cross-Attention ---
        query = self.cross_attn_skipconn(query, lambda x: self.cross_attn(x, feature, feat_mask))
        # --- FFN on Query ---
        query = self.query_ffn_skipconn(query, self.query_ffn)
        
        return query


class R2LDecoderLayer(nn.Module):

    def __init__(self, d_model: int, d_ff: int, multiple_of: int, num_heads: int, sublayer_num: int, dropout: float, use_rope: bool, first_layer: bool):
        assert sublayer_num == 3, "[R2LDecoderLayer.__init__] sublayer_num must be 3"
        super(R2LDecoderLayer, self).__init__()
        self.attn_1 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.attn_2 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.feed_forward = FFN(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, dropout=dropout)
        if first_layer:
            self.sublayer_connection_1 = SkipConnectionAfterLN(size=d_model, dropout=dropout)
        else:
            self.sublayer_connection_1 = SublayerConnection(size=d_model, dropout=dropout)
        self.sublayer_connection_2 = SublayerConnection(size=d_model, dropout=dropout)
        self.sublayer_connection_3 = SublayerConnection(size=d_model, dropout=dropout)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        if r2l_memory is not None:
            raise ValueError("[R2LDecoderLayer.forward] r2l_memory must be None")
        
        x = self.sublayer_connection_1(x, lambda x: self.attn_1(x, x, trg_mask))
        x = self.sublayer_connection_2(x, lambda x: self.attn_2(x, memory, src_mask))
        x = self.sublayer_connection_3(x, self.feed_forward)
        return x


class L2RDecoderLayer(nn.Module):

    def __init__(self, d_model: int, d_ff: int, multiple_of: int, num_heads: int, sublayer_num: int, dropout: float, use_rope: bool, first_layer: bool):
        assert sublayer_num == 4, "[L2RDecoderLayer.__init__] sublayer_num must be 4"
        super(L2RDecoderLayer, self).__init__()
        self.attn_1 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.attn_2 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.attn_3 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout, use_rope=use_rope)
        self.feed_forward = FFN(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, dropout=dropout)
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
            
        x = self.sublayer_connection_1(x, lambda x: self.attn_1(x, x, trg_mask))
        x = self.sublayer_connection_2(x, lambda x: self.attn_2(x, memory, src_mask))
        x = self.sublayer_connection_3(x, lambda x: self.attn_3(x, r2l_memory, r2l_trg_mask))
        x = self.sublayer_connection_4(x, self.feed_forward)
        return x


class Encoder(nn.Module):

    def __init__(self, d_model: int, d_ff: int, multiple_of: int, num_heads: int, num_layers: int, dropout: float, use_rope: bool):
        super(Encoder, self).__init__()
        self.src_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=num_heads, dropout=dropout, use_rope=use_rope,
                         first_layer=(i == 0))
            for i in range(num_layers)
        ])

    def forward(self, query, src, src_mask):
        src = self.src_norm(src)
        for layer in self.encoder_layers:
            query = layer(query, src, src_mask)
        query = self.query_norm(query)
        return query


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
    if not isinstance(src, tuple): 
        raise ValueError("[padmask] src must be a tuple of tensors for multi-feature input")
    
    enc_src_mask = tuple([(feat[:,:,0] != pad_idx).unsqueeze(1) for feat in src])
    dec_src_mask = enc_src_mask[0]
    for mask in enc_src_mask[1:]: dec_src_mask = dec_src_mask & mask # AND
    src_mask = (enc_src_mask, dec_src_mask)
    
    if trg is not None:
        sample_mask = dec_src_mask
        trg_mask = (trg != pad_idx).unsqueeze(1) & subsequent_mask(trg.size(1)).type_as(sample_mask.data)
        r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(sample_mask.data)
        r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(sample_mask.data)
        return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
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
                 n_enc_layers, n_dec_layers, dropout, device='cuda'):
        super(ABDTransformer, self).__init__()
        self.vocab  = vocab
        self.device = device
        multiple_of = 128
        num_queries = 10

        # --- Feature Embeddings ---
        self.r2l_visual_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
        self.r2l_motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
        self.r2l_imgcap_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
        
        self.l2r_visual_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
        self.l2r_motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
        self.l2r_imgcap_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
        
        # --- Text Embeddings ---
        self.r2l_trg_embed = TextEmbedding(vocab.n_vocabs, d_model)
        self.l2r_trg_embed = TextEmbedding(vocab.n_vocabs, d_model)
        self.pos_embed = PositionalEncoding(dim=d_model, dropout=dropout, max_len=256)
        
        # --- Encoders ---
        self.r2l_query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))
        self.r2l_visual_encoder_big = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads_big, num_layers=n_enc_layers, dropout=dropout, use_rope=False)
        self.r2l_motion_encoder_big = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads_big, num_layers=n_enc_layers, dropout=dropout, use_rope=False)
        self.r2l_imgcap_encoder_big = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads_big, num_layers=n_enc_layers, dropout=dropout, use_rope=False)

        self.l2r_query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))
        self.l2r_visual_encoder = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_enc_layers, dropout=dropout, use_rope=False)
        self.l2r_motion_encoder = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_enc_layers, dropout=dropout, use_rope=False)
        self.l2r_imgcap_encoder = Encoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_enc_layers, dropout=dropout, use_rope=False)

        # --- Decoders ---
        self.r2l_decoder = R2L_Decoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_dec_layers, dropout=dropout, use_rope=False)
        self.l2r_decoder = L2R_Decoder(d_model=d_model, d_ff=d_ff, multiple_of=multiple_of, num_heads=n_heads, num_layers=n_dec_layers, dropout=dropout, use_rope=False)

        # --- Generators ---
        self.r2l_generator = Generator(d_model=d_model, vocab_size=vocab.n_vocabs)
        self.l2r_generator = Generator(d_model=d_model, vocab_size=vocab.n_vocabs)

    def encode(self, src, src_mask, r2l_encode=False):
        # ============== Right-to-Left Encoding ==============
        if r2l_encode:
            # Expand queries to match batch size
            batch_size = src[0].size(0)
            queries = self.r2l_query_embed.repeat(batch_size, 1, 1)
            queries = self.pos_embed(queries)
            
            # --- Encode each feature type separately ---
            feat_1 = self.r2l_visual_src_embed(src[0])
            feat_1 = self.pos_embed(feat_1)
            queries = self.r2l_visual_encoder_big(queries, feat_1, src_mask[0])

            feat_2 = self.r2l_motion_src_embed(src[1])
            feat_2 = self.pos_embed(feat_2)
            queries = self.r2l_motion_encoder_big(queries, feat_2, src_mask[1])

            feat_3 = self.r2l_imgcap_src_embed(src[2])
            feat_3 = self.pos_embed(feat_3)
            queries = self.r2l_imgcap_encoder_big(queries, feat_3, src_mask[2])
            
            return queries
        
        # ============== Left-to-Right Encoding ==============
        else:
            # Expand queries to match batch size
            batch_size = src[0].size(0)
            queries = self.l2r_query_embed.repeat(batch_size, 1, 1)
            queries = self.pos_embed(queries)
            
            # --- Encode each feature type separately ---
            feat_1 = self.l2r_visual_src_embed(src[0])
            feat_1 = self.pos_embed(feat_1)
            queries = self.l2r_visual_encoder(queries, feat_1, src_mask[0])

            feat_2 = self.l2r_motion_src_embed(src[1])
            feat_2 = self.pos_embed(feat_2)
            queries = self.l2r_motion_encoder(queries, feat_2, src_mask[1])

            feat_3 = self.l2r_imgcap_src_embed(src[2])
            feat_3 = self.pos_embed(feat_3)
            queries = self.l2r_imgcap_encoder(queries, feat_3, src_mask[2])

            return queries

    def r2l_decode(self, r2l_trg, memory, src_mask, r2l_trg_mask):
        src_mask = None # ! Use all queries from encoder
        x = self.r2l_trg_embed(r2l_trg)
        x = self.pos_embed(x)
        return self.r2l_decoder(x, memory, src_mask, r2l_trg_mask)

    def l2r_decode(self, trg, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        src_mask = None # ! Use all queries from encoder
        x = self.l2r_trg_embed(trg)
        x = self.pos_embed(x)
        return self.l2r_decoder(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)

    def forward(self, src, r2l_trg, trg, mask):
        src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask = mask
        
        enc_src_mask, dec_src_mask = src_mask
        r2l_encoding_outputs = self.encode(src, enc_src_mask, r2l_encode=True)
        encoding_outputs = self.encode(src, enc_src_mask)
        
        r2l_outputs = self.r2l_decode(r2l_trg, r2l_encoding_outputs, dec_src_mask, r2l_trg_mask)
        l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

        r2l_pred = self.r2l_generator(r2l_outputs)
        l2r_pred = self.l2r_generator(l2r_outputs)
        
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
        r2l_model_encodings = self.encode(src, enc_src_mask, r2l_encode=True)
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
