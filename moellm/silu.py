import torch

from moellm.linear import Linear


def silu(in_features):
    return in_features * torch.nn.functional.sigmoid(in_features)


class SiLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))
