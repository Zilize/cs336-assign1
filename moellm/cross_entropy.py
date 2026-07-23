import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_max = logits.max(dim=-1, keepdim=True).values
    logits_shifted = logits - logits_max

    logits_exp = logits_shifted.exp()
    logits_exp_sum = logits_exp.sum(dim=-1, keepdim=True)
    logits_exp_sum_log = logits_exp_sum.log()
    logits_exp_sum_log = logits_exp_sum_log.reshape(-1)

    logits_shifted = logits_shifted.reshape(-1, logits_shifted.shape[-1])
    batch = torch.arange(logits_shifted.shape[0])
    logits_target = logits_shifted[batch, targets.reshape(-1)]

    return (-logits_target + logits_exp_sum_log).mean()
