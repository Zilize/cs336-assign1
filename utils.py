import os
import torch
import numpy as np
from tqdm import tqdm

from torch.distributed.tensor import DTensor

from cs336_basics.get_batch import get_batch


def dataloader(data_path, batch_size, context_len, is_valid=False, num_steps=None, device=None):
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


def capture_gradient_norms(model, num_layers, device=None, distributed=False):
    def capture_impl_local_(parameters):
        sum_of_squares = torch.zeros((1,), device=device)
        for parameter in parameters:
            if parameter.grad is None:
                continue
            sum_of_squares += (parameter.grad.data ** 2).sum()

    def capture_impl_dist_(parameters):
        norms = list()
        for parameter in parameters:
            if parameter.grad is None:
                continue
            norm = torch.linalg.vector_norm(parameter.grad.data.detach(), ord=2.0)
            norm = norm.full_tensor()
            norms.append(norm)
        stacked = torch.stack([x.flatten() for x in norms])
        return torch.linalg.vector_norm(stacked, 2.0)

    capture_impl_ = capture_impl_dist_ if distributed else capture_impl_local_
    return {
        "layers_first_attn_norm": capture_impl_(model.layers[0].attn.parameters()),
        "layers_first_ffn_norm": capture_impl_(model.layers[0].ffn.parameters()),
        "layers_last_attn_norm": capture_impl_(model.layers[num_layers - 1].attn.parameters()),
        "layers_last_ffn_norm": capture_impl_(model.layers[num_layers - 1].ffn.parameters())
    }


def capture_weight_norm(model, device=None, distributed=False):
    if distributed:
        norms = []
        for param in model.parameters():
            p = param.detach()
            n = torch.linalg.vector_norm(p.float(), ord=2.0)
            if isinstance(n, DTensor):
                n = n.full_tensor()
            norms.append(n.reshape(1))
        stacked = torch.cat(norms)
        return torch.linalg.vector_norm(stacked, ord=2.0).item()
    else:
        sum_of_squares = torch.zeros((1,), device=device)
        for parameter in model.parameters():
            sum_of_squares += (parameter.data.float() ** 2).sum()
        return torch.sqrt(sum_of_squares).item()


activation_rms = dict()

def capture_activation_rms_hook(layer_name, distributed=False):
    def hook(module, inputs, output):
        with torch.no_grad():
            sum_of_squares = output.detach().pow(2).sum()
            num_elements = torch.tensor(output.numel(), device=output.device, dtype=sum_of_squares.dtype)
            if distributed:
                torch.distributed.all_reduce(sum_of_squares, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(num_elements, op=torch.distributed.ReduceOp.SUM)
            activation_rms[layer_name] = (sum_of_squares / num_elements).sqrt().item()
    return hook
