from pathlib import Path
from typing import Protocol
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
CACHE_ROOT = PROJECT_ROOT / "cache"


class DatasetConfig(Protocol):
    dataset_name: str
    cache_dir: Path
    train_file: Path
    valid_file: Path
    vocab_size: int
    special_tokens: list[str]

    # 分段处理控制内存峰值，并发处理提升吞吐量
    num_iteration: int
    num_processes: int


@dataclass
class OWTConfig(DatasetConfig):
    dataset_name = "owt"
    cache_dir = CACHE_ROOT / dataset_name
    train_file = DATA_ROOT / "owt_train.txt"
    valid_file = DATA_ROOT / "owt_valid.txt"
    vocab_size = 32000
    special_tokens = ['<|endoftext|>']
    num_iteration = 100
    num_processes = 16


@dataclass
class TinyStoryConfig(DatasetConfig):
    dataset_name = "tinystory"
    cache_dir = CACHE_ROOT / dataset_name
    train_file = DATA_ROOT / "TinyStoriesV2-GPT4-train.txt"
    valid_file = DATA_ROOT / "TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']
    num_iteration = 10
    num_processes = 8
