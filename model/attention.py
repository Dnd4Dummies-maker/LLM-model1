import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.to(x_complex.device)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.wq = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.rope = config.rope
        if self.rope:
            self.register_buffer(
                "freqs_cis",
                precompute_rope_freqs(self.head_dim, config.max_seq_len),
                persistent=False,
            )

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope:
            freqs = self.freqs_cis[:T]
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.wo(out)
