import math
import torch

from typing import Callable


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(param))
                v = state.get("v", torch.zeros_like(param))

                param.data -= alpha * group["weight_decay"] * param.data

                m = betas[0] * m + (1 - betas[0]) * param.grad.data
                v = betas[1] * v + (1 - betas[1]) * (param.grad.data ** 2)

                alpha_t = alpha * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                param.data -= alpha_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
