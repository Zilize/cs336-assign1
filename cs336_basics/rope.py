import torch
from einops import rearrange


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        cos_matrix, sin_matrix = self.build_rotary_matrix()
        self.register_buffer('cos_matrix', cos_matrix, persistent=False)
        self.register_buffer('sin_matrix', sin_matrix, persistent=False)

    def build_rotary_matrix(self):
        k = torch.arange(1, self.d_k // 2 + 1)
        k = torch.repeat_interleave(k, repeats=2)
        p = (2 * k - 2) / self.d_k
        denominator = rearrange(torch.pow(self.theta, p), '... -> 1 ...')
        numerator = rearrange(torch.arange(0, self.max_seq_len), '... -> ... 1')
        angle = numerator / denominator  # (max_seq_len, d_k)
        return torch.cos(angle), torch.sin(angle)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        assert seq_len <= self.max_seq_len

        cos_matrix = self.cos_matrix[token_positions, ...]  # (seq_len, d_k)
        sin_matrix = self.sin_matrix[token_positions, ...]  # (seq_len, d_k)

        odd = torch.arange(1, self.d_k, 2)
        even = torch.arange(0, self.d_k, 2)
        indices = torch.stack([odd, even], dim=1).flatten()

        sign = torch.tensor([-1, 1]).repeat(self.d_k // 2)

        return x * cos_matrix + x[..., indices] * sign * sin_matrix
