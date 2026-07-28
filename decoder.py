import torch
import codecs
import argparse

from moellm.config import TinyStoryConfig
from moellm.softmax import softmax
from moellm.tokenizer import Tokenizer
from moellm.transformer import TransformerLM


END_OF_TEXT = b'<|endoftext|>'


def resolve_device(device: str) -> str:
    if device != 'auto':
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(args, device: str) -> TransformerLM:
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = payload['model'] if 'model' in payload else payload

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
    )
    lm.load_state_dict(state_dict)
    lm.to(device)
    lm.eval()
    return lm


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature == 0.0:
        return int(torch.argmax(logits))

    probs = softmax(logits / temperature, dim=-1)
    if top_p >= 1.0:
        return int(torch.multinomial(probs, num_samples=1))

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    keep = int((torch.cumsum(sorted_probs, dim=-1) < top_p).sum()) + 1
    sampled = torch.multinomial(sorted_probs[:keep], num_samples=1)
    return int(sorted_indices[sampled])


@torch.no_grad()
def generate(lm: TransformerLM, tokenizer: Tokenizer, args, device: str) -> None:
    tokens = tokenizer.encode(args.prompt)
    if not tokens:
        raise ValueError(f'prompt empty: {args.prompt!r}')
    eos_id = tokenizer.inverted_vocab.get(END_OF_TEXT)

    # 单个 BPE token 可能只是某个多字节字符的一部分，用增量解码器跨 token 拼接
    stream = codecs.getincrementaldecoder('utf-8')(errors='replace')
    print(args.prompt, end='', flush=True)

    for _ in range(args.max_new_tokens):
        window = torch.tensor(tokens[-args.context_len:], dtype=torch.long, device=device)
        next_token_logits = lm(window)[-1]
        next_token_id = sample_next_token(next_token_logits, args.temperature, args.top_p)

        if next_token_id == eos_id:
            break
        tokens.append(next_token_id)
        print(stream.decode(tokenizer.vocab[next_token_id]), end='', flush=True)

    print(stream.decode(b'', final=True))


def main(args):
    assert args.temperature >= 0.0
    assert 0.0 < args.top_p <= 1.0
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_files(
        TinyStoryConfig.cache_dir / 'vocab.pkl',
        TinyStoryConfig.cache_dir / 'merges.pkl',
        TinyStoryConfig.special_tokens)
    lm = load_model(args, device)
    generate(lm, tokenizer, args, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint/model.pt')
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--device', type=str, default='auto')

    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_len', type=int, default=256)

    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--num_heads', type=int, default=16)

    parser.add_argument('--use_flash_attn', action='store_true')
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--rope_theta', type=int, default=10000)
    main(parser.parse_args())
