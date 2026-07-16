import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhonobyteConfig:
    def __init__(self):
        self.vocab_size = 256
        self.max_seq_len = 128
        self.d_model = 256
        self.n_heads = 8
        self.n_layers = 4
        self.d_ff = 512
        self.dropout = 0.1

class DualChannelEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_embeddings = nn.Embedding(config.vocab_size, config.d_model // 2)
        self.lexicon_size = 10000
        self.id_embeddings = nn.Embedding(self.lexicon_size, config.d_model // 2)
        self.register_buffer('pe', self._build_positional_encoding(config.max_seq_len, config.d_model))

    def _build_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, mask_tensor, id_tensor):
        seq_len = mask_tensor.size(1)
        mask_features = self.mask_embeddings(mask_tensor)
        id_features = self.id_embeddings(id_tensor)
        fused_embeddings = torch.cat([mask_features, id_features], dim=-1)
        return fused_embeddings + self.pe[:, :seq_len]

class PhonobyteAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(embed_dim=config.d_model, num_heads=config.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, causal_mask=None):
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class RawPhonobyteTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = DualChannelEmbedding(config)
        self.layers = nn.ModuleList([PhonobyteAttentionBlock(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, mask_tensor, id_tensor):
        seq_len = mask_tensor.size(1)
        device = mask_tensor.device
        x = self.embedding(mask_tensor, id_tensor)
        causal_mask = self._generate_causal_mask(seq_len, device)
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits