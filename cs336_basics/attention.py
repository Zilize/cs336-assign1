import math
import torch
from einops import einsum, rearrange

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    query: torch.Tensor,  # (..., query_len, d_k)
    key: torch.Tensor,  # (..., key_len, d_k)
    value: torch.Tensor,  # (..., key_len, d_v),
    mask: torch.Tensor | None = None,  # (query_len, key_len)
):
    score = einsum(query, key, '... query_len d_k, ... key_len d_k -> ... query_len key_len')
    score = score / math.sqrt(query.shape[-1])

    if mask is not None:
        mask = ~mask
        score = torch.masked_fill(score, mask, float('-inf'))

    score = softmax(score, dim=-1)
    return einsum(score, value, '... query_len key_len, ... key_len d_v -> ... query_len d_v')


class MHA(torch.nn.Module):
    def __init__(self, d_model, num_heads, use_rope=False, rope_theta=None, rope_max_seq_len=None, use_flash_attn=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert self.d_head * self.num_heads == self.d_model
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn

        self.q_proj = Linear(self.d_model, self.d_model)
        self.k_proj = Linear(self.d_model, self.d_model)
        self.v_proj = Linear(self.d_model, self.d_model)
        self.output_proj = Linear(self.d_model, self.d_model)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(rope_theta, self.d_head, rope_max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, '... seq_len (h d_k) -> ... h seq_len d_k', h=self.num_heads)
        k = rearrange(k, '... seq_len (h d_k) -> ... h seq_len d_k', h=self.num_heads)
        v = rearrange(v, '... seq_len (h d_v) -> ... h seq_len d_v', h=self.num_heads)

        if self.use_rope:
            assert token_positions is not None
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool)).to(q.device)
        if self.use_flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
        else:
            output = scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(output, '... h seq_len d_v -> ... seq_len (h d_v)')

        return self.output_proj(output)
