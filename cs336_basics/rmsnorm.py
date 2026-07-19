import torch
from einops import rearrange


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_pow = torch.pow(x, 2)
        rms = torch.sqrt(torch.sum(x_pow, dim=-1) / self.d_model + self.eps)
        rms = rearrange(rms, '... -> ... 1')

        result = x / rms * self.weight
        return result.to(in_dtype)
