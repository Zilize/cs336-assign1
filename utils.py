import os
import torch
import numpy as np
from tqdm import tqdm

from cs336_basics.get_batch import get_batch


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def dataloader(data_path, batch_size, context_len, is_valid=False, num_steps=None):
    total_tokens = os.path.getsize(data_path) // 2
    dataset = np.memmap(data_path, dtype=np.uint16, mode='r', shape=(total_tokens,))

    if not is_valid:
        assert num_steps is not None
        for _ in tqdm(range(num_steps)):
            yield get_batch(dataset, batch_size, context_len, device)
    else:
        num_seqs = (total_tokens - 1) // context_len
        num_steps = num_seqs // batch_size
        for i in range(num_steps):
            input_start = np.arange(i * batch_size, (i + 1) * batch_size) * context_len
            yield get_batch(dataset, batch_size, context_len, device, input_start=input_start)

        input_start = np.arange(num_steps * batch_size, num_seqs) * context_len
        if num_seqs - num_steps * batch_size > 0:
            yield get_batch(dataset, num_seqs - num_steps * batch_size, context_len, device, input_start)


def capture_gradient_norms(named_parameters, num_layers):
    layers_first_attn_sum_of_squares = torch.zeros((1,), device=device)
    layers_first_ffn_sum_of_squares = torch.zeros((1,), device=device)
    layers_last_attn_sum_of_squares = torch.zeros((1,), device=device)
    layers_last_ffn_sum_of_squares = torch.zeros((1,), device=device)
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        if name.startswith(f"layers.0.attn"):
            layers_first_attn_sum_of_squares += (parameter.grad.data ** 2).sum()
        elif name.startswith(f"layers.0.ffn"):
            layers_first_ffn_sum_of_squares += (parameter.grad.data ** 2).sum()
        elif name.startswith(f"layers.{num_layers - 1}.attn"):
            layers_last_attn_sum_of_squares += (parameter.grad.data ** 2).sum()
        elif name.startswith(f"layers.{num_layers - 1}.ffn"):
            layers_last_ffn_sum_of_squares += (parameter.grad.data ** 2).sum()
    return {
        "layers_first_attn_norm": torch.sqrt(layers_first_attn_sum_of_squares).item(),
        "layers_first_ffn_norm": torch.sqrt(layers_first_ffn_sum_of_squares).item(),
        "layers_last_attn_norm": torch.sqrt(layers_last_attn_sum_of_squares).item(),
        "layers_last_ffn_norm": torch.sqrt(layers_last_ffn_sum_of_squares).item()
    }


def capture_weight_norm(parameters):
    sum_of_squares = torch.zeros((1,), device=device)
    for parameter in parameters:
        sum_of_squares += (parameter.data ** 2).sum()
    return torch.sqrt(sum_of_squares).item()


activation_norms = dict()

def capture_activation_norm_hook(layer_name):
    def hook(module, inputs, output):
        with torch.no_grad():
            norm = output.detach().norm().item()
            activation_norms[layer_name] = norm
    return hook
