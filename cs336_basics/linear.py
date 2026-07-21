import math
import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))

        std = math.sqrt(2. / (in_features + out_features))
        truncation = 3. * std
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-truncation, b=truncation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')
