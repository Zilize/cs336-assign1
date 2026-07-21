import torch


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.weight[token_ids]
