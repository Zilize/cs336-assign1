import torch
from einops import einsum

from cs336_basics.attention import MHA
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.softmax import softmax
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, use_rope=True, rope_theta=None, rope_max_seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attn = MHA(
            self.d_model,
            self.num_heads,
            use_rope=use_rope,
            rope_theta=rope_theta,
            rope_max_seq_len=rope_max_seq_len)
        self.ffn = SwiGLU(self.d_model, self.d_ff)

        self.ln1 = RMSNorm(self.d_model)
        self.ln2 = RMSNorm(self.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(0, x.shape[-2])
        x = self.attn(self.ln1(x), token_positions) + x
        x = self.ffn(self.ln2(x)) + x
        return x


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff, use_rope=True,
                 rope_theta=None, rope_max_seq_len=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.token_embeddings = Embedding(self.vocab_size, d_model)
        self.layers = torch.nn.ModuleList([TransformerBlock(
            d_model,
            num_heads,
            d_ff,
            use_rope=use_rope,
            rope_theta=rope_theta,
            rope_max_seq_len=rope_max_seq_len,
        ) for _ in range(self.num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Embedding(self.vocab_size, d_model)

    def forward(self, in_indices: torch.LongTensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)

        output = einsum(x, self.lm_head.weight, 'batch_size seq_len d_model, vocab_size d_model -> '
                                                'batch_size seq_len vocab_size')
        return output
