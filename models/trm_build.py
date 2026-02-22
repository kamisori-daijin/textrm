import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """
    ANE-optimized RMSNorm using LayerNorm trick
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        doubled = torch.cat([x, -x], dim=-1)

        normed = F.layer_norm(
            doubled,
            normalized_shape=(doubled.shape[-1],),
            weight=None,
            bias=None,
            eps=self.eps,
        )

        normed = normed[..., : x.shape[-1]]
        return normed * self.weight


# ============================================================================
# Rotary Embedding (unchanged, CoreML-safe)
# ============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x):
        seq_len = x.shape[-1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=1)
    return torch.cat([-x2, x1], dim=1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q/k: [B, H, Hd, 1, T]
    # cos, sin: [T, Hd]
    # Need to reshape cos/sin to [1, 1, Hd, 1, T] for broadcasting
    cos = cos.transpose(0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(3)  # [1, 1, Hd, 1, T]
    sin = sin.transpose(0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(3)  # [1, 1, Hd, 1, T]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# ============================================================================
# SwiGLU (Conv2d)
# ============================================================================

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.w2 = nn.Conv2d(hidden_dim, dim, 1, bias=False)
        self.w3 = nn.Conv2d(dim, hidden_dim, 1, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================================
# Causal Self Attention (linear)
# ============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE"""
    def __init__(self, dim, n_heads, max_seq_len=512):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# ============================================================================
# Transformer Block (ANE)
# ============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, max_seq_len=512):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # to Conv2d layout
        y = self.norm2(x).permute(0, 2, 1).unsqueeze(2)
        y = self.mlp(y).squeeze(2).permute(0, 2, 1)
        return x + y
