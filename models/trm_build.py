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
# Causal Self Attention (ANE Conv2d)
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len=512):
        super().__init__()
        assert dim % n_heads == 0

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.o_proj = nn.Conv2d(dim, dim, 1, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        # x: [B, T, D] → [B, D, 1, T]
        B, T, D = x.shape
        x = x.permute(0, 2, 1).unsqueeze(2)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, H, Hd, 1, T]
        q = q.view(B, self.n_heads, self.head_dim, 1, T)
        k = k.view(B, self.n_heads, self.head_dim, 1, T)
        v = v.view(B, self.n_heads, self.head_dim, 1, T)

        cos, sin = self.rope(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        att = torch.matmul(q.transpose(-1, -2), k) * self.scale
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, v.transpose(-1, -2))
        out = out.reshape(B, D, 1, T)

        out = self.o_proj(out)

        # back to [B, T, D]
        return out.squeeze(2).permute(0, 2, 1)


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
