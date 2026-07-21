import torch
import numpy as np
import numpy.typing as npt


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str, input_start=None):
    dataset_length = dataset.size

    if input_start is None:
        input_start = np.random.randint(0, dataset_length - context_length, size=batch_size)
    else:
        assert len(input_start) == batch_size
    input_index = input_start[:, None] + np.arange(context_length)
    output_index = input_index + 1

    input_tensor = torch.tensor(dataset[input_index], dtype=torch.long, device=device)
    output_tensor = torch.tensor(dataset[output_index], dtype=torch.long, device=device)
    return input_tensor, output_tensor
