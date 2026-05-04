import pickle as pkl
from typing import Iterable, Iterator

from cs336_basics.config import TinyStoryConfig
from cs336_basics.tokenizer.utils import pre_tokenize_from_text


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

        # build merge_rank from merge
        self.merge_rank: dict[tuple[bytes, bytes], int] = dict()
        for rank, merge in enumerate(self.merges):
            self.merge_rank[merge] = rank

        self.inverted_vocab = {self.vocab[key]: key for key in self.vocab}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        with open(vocab_filepath, 'rb') as file:
            vocab = pkl.load(file)
        with open(merges_filepath, 'rb') as file:
            merges = pkl.load(file)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        assert self.special_tokens is not None
        pre_tokens = pre_tokenize_from_text(text, self.special_tokens)

        merged_pre_tokens = list()
        for pre_token in pre_tokens:
            while len(pre_token) > 1:
                pairs = [(pre_token[i], pre_token[i + 1]) for i in range(len(pre_token) - 1)]
                merge_candidate = min(pairs, key=lambda p: self.merge_rank.get(p, float("inf")))

                if merge_candidate not in self.merge_rank:
                    break

                i = 0
                new_pre_token_list = list()
                while i < len(pre_token):
                    if i < len(pre_token) - 1 and (pre_token[i], pre_token[i + 1]) == merge_candidate:
                        merged = merge_candidate[0] + merge_candidate[1]
                        new_pre_token_list.append(merged)
                        i += 2
                    else:
                        new_pre_token_list.append(pre_token[i])
                        i += 1

                pre_token = tuple(new_pre_token_list)

            merged_pre_tokens.append(pre_token)

        result = list()
        for merged_pre_token in merged_pre_tokens:
            result.extend([self.vocab[item] for item in merged_pre_token])
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            token_ids = self.encode(it)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        result = bytes()
        for token_id in ids:
            result += self.inverted_vocab[token_id]
        return result.decode()


def main():
    tokenizer = Tokenizer.from_files(
        TinyStoryConfig.cache_dir / "vocab.pkl",
        TinyStoryConfig.cache_dir / "merges.pkl",
        TinyStoryConfig.special_tokens)
    ids = tokenizer.encode("Hello, World!")
    print(tokenizer.decode(ids))


if __name__ == '__main__':
    main()
