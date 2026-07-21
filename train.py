import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from cs336_basics.adamw import AdamW
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.get_batch import get_batch
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import learning_rate_schedule
from cs336_basics.transformer import TransformerLM


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def dataloader(data_path, step_count, batch_size, context_len):
    total_tokens = os.path.getsize(data_path) // 2
    dataset = np.memmap(data_path, dtype=np.uint16, mode='r', shape=(total_tokens,))

    for _ in tqdm(range(step_count)):
        yield get_batch(dataset, batch_size, context_len, device)


def train(args):
    lm = TransformerLM(
        args.vocab_size,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.use_rope,
        args.rope_theta,
        args.context_len,
        args.use_flash_attn
    ).to(device)

    optimizer = AdamW(
        lm.parameters(),
        lr=0.0,
        weight_decay=args.weight_decay,
        betas=(args.beta_1, args.beta_2)
    )

    iteration = 0
    for inputs, targets in dataloader(args.train_data, args.max_iterations, args.batch_size, args.context_len):
        lm.zero_grad()
        outputs = lm(inputs)

        loss = cross_entropy(outputs, targets)
        loss.backward()

        gradient_clipping(lm.parameters(), max_l2_norm=args.max_l2_norm, device=device)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate_schedule(
                iteration,
                args.max_learning_rate,
                args.min_learning_rate,
                args.warmup_iters,
                args.cosine_cycle_iters
            )
        optimizer.step()

        print(loss.item())

        if iteration % args.save_intervals == 0:
            save_checkpoint(lm, optimizer, iteration, f'checkpoint/{iteration}.pt')
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='./cache/tinystory/train_encoded.bin')
    parser.add_argument('--valid_data', type=str, default='./cache/tinystory/valid_encoded.bin')

    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--context_len', type=int, default=256)

    parser.add_argument('--max_iterations', type=int, default=100000)
    parser.add_argument('--save_intervals', type=int, default=100)

    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--num_heads', type=int, default=16)

    parser.add_argument('--use_rope', type=bool, default=True)
    parser.add_argument('--rope_theta', type=int, default=10000)
    parser.add_argument('--use_flash_attn', type=bool, default=True)

    parser.add_argument('--max_learning_rate', type=float, default=1e-2)
    parser.add_argument('--min_learning_rate', type=float, default=1e-3)
    parser.add_argument('--warmup_iters', type=int, default=2000)
    parser.add_argument('--cosine_cycle_iters', type=int, default=10000)

    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=int, default=0.999)

    parser.add_argument('--max_l2_norm', type=float, default=1.0)
    train(parser.parse_args())
