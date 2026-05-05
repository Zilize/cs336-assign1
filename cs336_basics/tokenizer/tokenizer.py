import os
import array
import pickle as pkl
from tqdm import tqdm
import multiprocessing as mp
from collections.abc import Iterable, Iterator

from cs336_basics.config import TinyStoryConfig
from cs336_basics.tokenizer.utils import pre_tokenize_from_text, find_nested_chunk_boundaries


mp.set_start_method("spawn", force=True)


class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
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

    @staticmethod
    def _encode_impl_from_text(
            text: str,
            special_tokens: list[str],
            merge_rank: dict[tuple[bytes, bytes], int],
            inverted_vocab: dict[bytes, int]
    ) -> list[int]:
        pre_tokens = pre_tokenize_from_text(text, special_tokens)

        merged_pre_tokens = list()
        for pre_token in pre_tokens:
            while len(pre_token) > 1:
                pairs = [(pre_token[i], pre_token[i + 1]) for i in range(len(pre_token) - 1)]
                merge_candidate = min(pairs, key=lambda p: merge_rank.get(p, float("inf")))
                if merge_candidate not in merge_rank:
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
            result.extend([inverted_vocab[item] for item in merged_pre_token])
        return result

    @staticmethod
    def _encode_impl_from_file(
            input_path: str | os.PathLike[str],
            start: int,
            end: int,
            special_tokens: list[str],
            merge_rank: dict[tuple[bytes, bytes], int],
            inverted_vocab: dict[bytes, int]
    ):
        with open(input_path, "rb") as f:
            f.seek(start)
            input_text = f.read(end - start).decode("utf-8", errors="ignore")
        return Tokenizer._encode_impl_from_text(input_text, special_tokens, merge_rank, inverted_vocab)

    def encode(self, text: str) -> list[int]:
        return self._encode_impl_from_text(
            text,
            self.special_tokens,
            self.merge_rank,
            self.inverted_vocab
        )

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            token_ids = self.encode(it)
            yield from token_ids

    def encode_from_file(
            self,
            input_path: str | os.PathLike[str],
            output_path: str | os.PathLike[str],
            num_iterations: int = 10,
            num_processes: int = 10,
    ):
        boundaries_list = find_nested_chunk_boundaries(input_path, num_iterations, num_processes)

        with open(output_path, 'wb') as output_file:
            for boundaries in tqdm(boundaries_list):
                args = list()
                for boundary in boundaries:
                    args.append((
                        input_path,
                        boundary[0],
                        boundary[1],
                        self.special_tokens,
                        self.merge_rank,
                        self.inverted_vocab
                    ))

                with mp.Pool(processes=num_processes) as pool:
                    results = pool.starmap(self._encode_impl_from_file, args)
                    for result in results:
                        array.array("H", result).tofile(output_file)

    def decode(self, ids: list[int]) -> str:
        result = b''
        for token_id in ids:
            result += self.vocab[token_id]
        return result.decode(errors='replace')

    @staticmethod
    def _decode_impl_from_file(
            input_path: str | os.PathLike[str],
            start: int,
            end: int,
            vocab: dict[int, bytes]
    ) -> bytes:
        with open(input_path, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
        ids = array.array("H", data).tolist()

        result = b''
        for token_id in ids:
            result += vocab[token_id]
        # 返回 bytes 直接以 wb 形式写入文件规避编码冲突
        return result

    def decode_from_file(
            self,
            input_path: str | os.PathLike[str],
            output_path: str | os.PathLike[str],
            num_iterations: int = 10,
            num_processes: int = 10,
    ):
        with open(input_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
        assert file_size % 2 == 0

        major_chunk_size = (file_size // 2 // num_iterations) * 2
        major_boundaries = [i * major_chunk_size for i in range(num_iterations)]
        major_boundaries.append(file_size)

        boundaries_list: list[list[tuple[int, int]]] = list()
        for start, end in zip(major_boundaries[:-1], major_boundaries[1:]):
            minor_chunk_size = ((end - start) // 2 // num_processes) * 2
            minor_boundaries = [start + i * minor_chunk_size for i in range(num_processes)]
            minor_boundaries.append(end)
            boundaries_list.append(list(zip(minor_boundaries[:-1], minor_boundaries[1:])))

        with open(output_path, 'wb') as output_file:
            for boundaries in tqdm(boundaries_list):
                args = list()
                for boundary in boundaries:
                    args.append((
                        input_path,
                        boundary[0],
                        boundary[1],
                        self.vocab
                    ))

                with mp.Pool(processes=num_processes) as pool:
                    results = pool.starmap(self._decode_impl_from_file, args)
                    for result in results:
                        output_file.write(result)


def tokenize_dataset():
    tokenizer = Tokenizer.from_files(
        TinyStoryConfig.cache_dir / "vocab.pkl",
        TinyStoryConfig.cache_dir / "merges.pkl",
        TinyStoryConfig.special_tokens)
    tokenizer.encode_from_file(
        TinyStoryConfig.valid_file,
        TinyStoryConfig.cache_dir / "valid_encoded.bin"
    )
    tokenizer.decode_from_file(
        TinyStoryConfig.cache_dir / "valid_encoded.bin",
        TinyStoryConfig.cache_dir / "valid_decoded.txt"
    )


if __name__ == '__main__':
    tokenize_dataset()
