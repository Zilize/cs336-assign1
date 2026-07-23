import torch


def gradient_clipping(parameters, max_l2_norm: float, eps=1e-6, device=None):
    parameters = list(filter(lambda x: x.grad is not None, parameters))

    sum_of_squares = torch.zeros((1,), device=device)
    for param in parameters:
        if param.grad is None:
            continue
        sum_of_squares += (param.grad.data ** 2).sum()
    l2_norm = torch.sqrt(sum_of_squares)

    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.data.mul_(scale)
    return l2_norm.item()
