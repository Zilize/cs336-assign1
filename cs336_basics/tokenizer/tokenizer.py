import pickle as pkl
from typing import Iterable, Iterator


class Tokenizer:
    def __init__(
            self,
            vocab: dict[bytes, int],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None=None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        with open(vocab_filepath, 'rb') as file:
            vocab = pkl.load(file)
        with open(merges_filepath, 'rb') as file:
            merges = pkl.load(file)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass
