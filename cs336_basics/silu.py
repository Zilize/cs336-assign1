import torch


def silu(in_features):
    return in_features * torch.nn.functional.sigmoid(in_features)
