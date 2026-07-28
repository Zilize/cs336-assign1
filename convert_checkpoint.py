"""Convert a DCP (torch.distributed.checkpoint) directory saved by train_distributed.py
into a plain state dict that can be loaded with TransformerLM.load_state_dict().
"""

import argparse
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


MODEL_PREFIX = "app.model."
ITER_KEY = "app.iter"

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def find_checkpoints(root: Path) -> list[Path]:
    """A checkpoint dir is one that directly contains a `.metadata` file. Training writes
    to `checkpoint/{iteration}`, but a single checkpoint dir may also be passed directly.
    """
    if (root / ".metadata").is_file():
        return [root]

    candidates = [d for d in root.iterdir() if d.is_dir() and (d / ".metadata").is_file()]

    def sort_key(path: Path):
        return (0, int(path.name)) if path.name.isdigit() else (1, path.name)

    return sorted(candidates, key=sort_key)


def load_dcp_model_state(checkpoint_dir: Path) -> tuple[dict[str, torch.Tensor], int | None]:
    state_dict: dict = {}
    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(keys=[MODEL_PREFIX.rstrip("."), ITER_KEY]),
        no_dist=True,
    )

    flat = {}

    def flatten(prefix: str, value):
        if isinstance(value, dict):
            for k, v in value.items():
                flatten(f"{prefix}{k}.", v)
        else:
            flat[prefix[:-1]] = value

    flatten("", state_dict)

    model_state = {
        key[len(MODEL_PREFIX):]: value
        for key, value in flat.items()
        if key.startswith(MODEL_PREFIX)
    }
    if not model_state:
        raise RuntimeError(f"no keys starting with '{MODEL_PREFIX}' found in {checkpoint_dir}")

    iteration = flat.get(ITER_KEY)
    if isinstance(iteration, torch.Tensor):
        iteration = int(iteration.item())
    return model_state, iteration


def convert(checkpoint_dir: Path, output_path: Path, dtype: torch.dtype, with_metadata: bool) -> None:
    model_state, iteration = load_dcp_model_state(checkpoint_dir)
    model_state = {
        key: value.to(dtype).contiguous() if value.is_floating_point() else value.contiguous()
        for key, value in model_state.items()
    }

    payload = model_state
    if with_metadata:
        payload = {"model": model_state, "iteration": iteration, "source": str(checkpoint_dir)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    num_params = sum(t.numel() for t in model_state.values())
    size_mb = sum(t.numel() * t.element_size() for t in model_state.values()) / 1024 ** 2
    print(
        f"{checkpoint_dir} -> {output_path} "
        f"(iteration={iteration}, tensors={len(model_state)}, params={num_params / 1e6:.2f}M, {size_mb:.1f}MiB)"
    )


def main(args) -> None:
    root = Path(args.checkpoint_dir)
    checkpoints = find_checkpoints(root)
    if not checkpoints:
        raise FileNotFoundError(f"no DCP checkpoint (directory with a .metadata file) under {root}")

    if args.list:
        for path in checkpoints:
            print(path)
        return

    if args.step is not None:
        selected = [p for p in checkpoints if p.name == str(args.step)]
        if not selected:
            raise ValueError(f"step {args.step} not found, available: {[p.name for p in checkpoints]}")
    elif args.all:
        selected = checkpoints
    else:
        selected = [checkpoints[-1]]

    dtype = DTYPES[args.dtype]
    output = Path(args.output)
    for checkpoint_dir in selected:
        if len(selected) > 1 or output.is_dir() or output.suffix == "":
            output_path = output / f"{checkpoint_dir.name}.pt"
        else:
            output_path = output
        convert(checkpoint_dir, output_path, dtype, args.with_metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint",
                        help="a DCP checkpoint dir, or a parent dir holding `{iteration}` subdirs")
    parser.add_argument("--output", type=str, default="checkpoint/model.pt",
                        help="output file, or a directory when converting multiple checkpoints")
    parser.add_argument("--step", type=int, default=None, help="iteration to convert, defaults to the latest")
    parser.add_argument("--all", action="store_true", help="convert every checkpoint found")
    parser.add_argument("--list", action="store_true", help="list available checkpoints and exit")
    parser.add_argument("--dtype", type=str, default="float32", choices=list(DTYPES))
    parser.add_argument("--with_metadata", action="store_true",
                        help="wrap the weights as {'model': ..., 'iteration': ...} instead of a bare state dict")
    main(parser.parse_args())
