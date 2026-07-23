import torch

from moellm.linear import Linear


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        gate = gate * torch.nn.functional.sigmoid(gate)

        signal = self.w3(x)
        gated_signal = gate * signal

        return self.w2(gated_signal)
