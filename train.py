import math
import torch
import wandb
import argparse

from cs336_basics.adamw import AdamW
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import learning_rate_schedule
from cs336_basics.transformer import TransformerLM
from utils import dataloader
from utils import capture_weight_norm, capture_gradient_norms, capture_activation_norm_hook, activation_norms


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def train(args):
    run = wandb.init(
        entity='zilize',
        project='llm',
        config=dict(vars(args))
    )
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

    lm.layers[0].register_forward_hook(capture_activation_norm_hook('layers.0'))
    lm.layers[args.num_layers - 1].register_forward_hook(capture_activation_norm_hook(f'layers.{args.num_layers - 1}'))

    optimizer = AdamW(
        lm.parameters(),
        lr=0.0,
        weight_decay=args.weight_decay,
        betas=(args.beta_1, args.beta_2)
    )

    iteration = 0
    for inputs, targets in dataloader(
            args.train_data,
            args.batch_size,
            args.context_len,
            num_steps=args.max_iterations,
            device=device
    ):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = lm(inputs)
            loss = cross_entropy(outputs, targets)

        loss.backward()
        gradient_norms = capture_gradient_norms(lm, args.num_layers, device=device)
        gradient_global_norm = gradient_clipping(lm.parameters(), max_l2_norm=args.max_l2_norm, device=device)

        learning_rate = learning_rate_schedule(
                iteration,
                args.max_learning_rate,
                args.min_learning_rate,
                args.warmup_iters,
                args.cosine_cycle_iters
            )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        lm.zero_grad()

        run.log({
            "train/loss": loss.item(),
            "train/learning_rate": learning_rate,
            "gradient_norm/global": gradient_global_norm,
            "gradient_norm/layers.0.attn": gradient_norms["layers_first_attn_norm"],
            "gradient_norm/layers.0.ffn": gradient_norms["layers_first_ffn_norm"],
            f"gradient_norm/layers.{args.num_layers - 1}.attn": gradient_norms["layers_last_attn_norm"],
            f"gradient_norm/layers.{args.num_layers - 1}.ffn": gradient_norms["layers_last_ffn_norm"],
            "activation_norm/layers.0": activation_norms["layers.0"],
            f"activation_norm/layers.{args.num_layers - 1}": activation_norms[f"layers.{args.num_layers - 1}"],
        }, step=iteration)

        if iteration % args.eval_intervals == 0:
            with torch.no_grad():
                lm.eval()
                total_valid_loss = 0
                total_valid_step = 0
                for valid_inputs, valid_targets in dataloader(
                        args.valid_data,
                        args.batch_size,
                        args.context_len,
                        is_valid=True,
                        device=device
                ):
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        valid_outputs = lm(valid_inputs)
                        valid_loss = cross_entropy(valid_outputs, valid_targets)

                    valid_batch_size = valid_inputs.shape[0]
                    total_valid_loss += valid_loss.item() * valid_batch_size
                    total_valid_step += valid_batch_size

                mean_valid_loss = total_valid_loss / total_valid_step
                weight_norm = capture_weight_norm(lm.parameters(), device=device)
                run.log({
                    "valid/loss": mean_valid_loss,
                    "valid/perplexity": math.exp(mean_valid_loss),
                    "valid/weight_norm": weight_norm,
                }, step=iteration)
                lm.train()

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
    parser.add_argument('--eval_intervals', type=int, default=100)
    parser.add_argument('--save_intervals', type=int, default=1000)

    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--num_heads', type=int, default=16)

    parser.add_argument('--use_flash_attn', action='store_true')
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--rope_theta', type=int, default=10000)

    parser.add_argument('--max_learning_rate', type=float, default=1e-2)
    parser.add_argument('--min_learning_rate', type=float, default=1e-3)
    parser.add_argument('--warmup_iters', type=int, default=2000)
    parser.add_argument('--cosine_cycle_iters', type=int, default=10000)

    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)

    parser.add_argument('--max_l2_norm', type=float, default=1.0)
    train(parser.parse_args())
