import os
import pickle as pkl
from tqdm import tqdm
import multiprocessing as mp
from sortedcontainers import SortedSet

from cs336_basics.config import DatasetConfig, TinyStoryConfig, OWTConfig
from cs336_basics.tokenizer.utils import find_nested_chunk_boundaries, pre_tokenize_from_file


mp.set_start_method("spawn", force=True)


class TokenizerBuilder:
    def __init__(
        self,
        input_path: str | os.PathLike[str],
        vocab_size: int,
        special_tokens: list[str],
        cache_dir: str | os.PathLike[str] | None = None,
        num_iterations: int = 10,
        num_processes: int = 20,
    ):
        # 固定使用了 16 位编码，对词表大小有限制
        assert vocab_size <= 65535
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.cache_dir = cache_dir

        self.num_iterations = num_iterations
        self.num_processes = num_processes

        self.vocab: dict[int, bytes] = self._init_vocab()
        assert vocab_size >= len(self.vocab)
        self.merges: list[tuple[bytes, bytes]] = list()

        self.pre_tokens: dict[tuple[bytes, ...], int] | None = None

    def _init_vocab(self) -> dict[int, bytes]:
        vocabulary = {i: bytes([i]) for i in range(256)}
        for special_token in self.special_tokens:
            vocabulary[len(vocabulary)] = special_token.encode('utf-8')
        return vocabulary

    @staticmethod
    def _merge(
            pre_tokens: dict[tuple[bytes, ...], int],
            frequency: dict[tuple[bytes, bytes], int],
            sorted_pairs: SortedSet,
            inverted: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
            merge_candidate: tuple[bytes, bytes]
    ):
        queue = list(inverted[merge_candidate])
        for pre_token in queue:
            new_pre_token_list: list[bytes] = list()

            i = 0
            subtract_index_pair_set = set()  # index in pre_token
            add_index_set: set[int] = set()  # index in new_pre_token_list

            while i < len(pre_token):
                if i < len(pre_token) - 1 and (pre_token[i], pre_token[i + 1]) == merge_candidate:
                    merged = merge_candidate[0] + merge_candidate[1]
                    new_pre_token_list.append(merged)

                    # 发生合并，需要考虑对前后 pair 的影响
                    # (pre_token[i - 1], pre_token[i]) 的频次转变为 (pre_token[i - 1], merged) 的频次
                    # (pre_token[i + 1], pre_token[i + 2]) 的频次转变为 (merged, pre_token[i + 2]) 的频次
                    # 但由于需要考虑连续的情况，所以采用先收集后处理的方式

                    # 发生合并，处理词频的减少，先收集去重避免连续情况下重复统计
                    # 比如 (b'i', b'n', b'i', b'n') -> (b"in", b"in")
                    if i - 1 >= 0:
                        front_index_pair = (i - 1, i)
                        subtract_index_pair_set.add(front_index_pair)
                    if i + 2 <= len(pre_token) - 1:
                        back_index_pair = (i + 1, i + 2)
                        subtract_index_pair_set.add(back_index_pair)
                    # 发生合并，处理词频的新增，先收集后处理
                    add_index_set.add(len(new_pre_token_list) - 1)

                    i += 2
                else:
                    new_pre_token_list.append(pre_token[i])
                    i += 1

            new_pre_token = tuple(new_pre_token_list)

            # 1. 增量更新 frequency
            # 合并后消失的共现组，从频率表中减去；
            for subtract_index_pair in subtract_index_pair_set:
                subtract_item = (pre_token[subtract_index_pair[0]], pre_token[subtract_index_pair[1]])
                previous_sorted_pair = (frequency[subtract_item], subtract_item)
                frequency[subtract_item] -= pre_tokens[pre_token]
                sorted_pairs.remove(previous_sorted_pair)

                if frequency[subtract_item] == 0:
                    del frequency[subtract_item]
                else:
                    next_sorted_pair = (frequency[subtract_item], subtract_item)
                    sorted_pairs.add(next_sorted_pair)

            # 合并后新增的元素，构建共现组再调整频率表
            add_index_pair_set = set()
            for add_index in add_index_set:
                if add_index > 0:
                    add_index_pair = (add_index - 1, add_index)
                    add_index_pair_set.add(add_index_pair)
                if add_index + 1 <= len(new_pre_token_list) - 1:
                    add_index_pair = (add_index, add_index + 1)
                    add_index_pair_set.add(add_index_pair)
            for add_index_pair in add_index_pair_set:
                # PyCharm 静态分析有点问题，故拆开来写
                a: int = add_index_pair[0]
                b: int = add_index_pair[1]
                add_item = (new_pre_token_list[a], new_pre_token_list[b])
                if add_item in frequency:
                    previous_sorted_pair = (frequency[add_item], add_item)
                    frequency[add_item] += pre_tokens[pre_token]
                    sorted_pairs.remove(previous_sorted_pair)

                    next_sorted_pair = (frequency[add_item], add_item)
                    sorted_pairs.add(next_sorted_pair)
                else:
                    frequency[add_item] = pre_tokens[pre_token]
                    sorted_pairs.add((frequency[add_item], add_item))

            # 2. 统一更新倒排索引表
            old_pair_set, new_pair_set = set(), set()
            for i in range(len(pre_token) - 1):
                old_pair_set.add((pre_token[i], pre_token[i + 1]))
            for i in range(len(new_pre_token) - 1):
                new_pair_set.add((new_pre_token[i], new_pre_token[i + 1]))
            removed_pairs = old_pair_set - new_pair_set
            for removed_pair in removed_pairs:
                inverted[removed_pair].discard(pre_token)
            # 对 new_pre_token 的所有 pair：
            # 如果 pair 是已有的，把 pre_token 替换为 new_pre_token
            # 如果 pair 是新增的，新增 new_pre_token
            for pair in new_pair_set:
                inverted.setdefault(pair, set()).discard(pre_token)
                inverted.setdefault(pair, set()).add(new_pre_token)

            # 3. 就地修改 pre_tokens
            # 如果发生合并，才考虑换 key
            new_pre_token_count = pre_tokens[pre_token]
            del pre_tokens[pre_token]
            pre_tokens[new_pre_token] = new_pre_token_count

        previous_sorted_pair = (frequency[merge_candidate], merge_candidate)
        sorted_pairs.remove(previous_sorted_pair)
        del frequency[merge_candidate]
        del inverted[merge_candidate]
        return pre_tokens, frequency

    def _pre_tokenize_parallel(self) -> dict[tuple[bytes, ...], int]:
        boundaries_list = find_nested_chunk_boundaries(self.input_path, self.num_iterations, self.num_processes)

        pre_tokens = dict()
        with mp.Pool(processes=self.num_processes) as pool:
            for boundaries in tqdm(boundaries_list):
                args = list()
                for boundary in boundaries:
                    args.append((
                        self.input_path,
                        boundary[0],
                        boundary[1],
                        self.special_tokens
                    ))
                results = pool.starmap(pre_tokenize_from_file, args)
                for result in results:
                    for key in result:
                        pre_tokens[key] = pre_tokens.get(key, 0) + result[key]
        return pre_tokens

    def _bpe_merge(self, pre_tokens: dict[tuple[bytes, ...], int]) -> None:
        # 要增量式维护两个数据结构：
        # frequency: pair 的频率统计表，用于找到频率最大的 pair；
        # sorted_pairs: frequency 用来找最大值太慢了，所以加一个辅助数据结构以便在 log(n) 内找到最值
        # inverted: pair 的倒排索引，用于快速找到合并所影响到的 pre_token
        frequency: dict[tuple[bytes, bytes], int] = dict()
        inverted: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = dict()
        for pre_token in tqdm(pre_tokens):
            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i + 1])
                frequency[pair] = frequency.get(pair, 0) + pre_tokens[pre_token]
                inverted.setdefault(pair, set()).add(pre_token)
        sorted_pairs = SortedSet()  # element type: tuple[int, tuple[bytes, bytes]]
        for pair in frequency:
            sorted_pairs.add((frequency[pair], pair))

        with tqdm(total=self.vocab_size, initial=len(self.vocab), desc="BPE training") as pbar:
            while len(self.vocab) < self.vocab_size:
                if len(frequency) <= 0:
                    # 对于小数据集，可能达成每个 pre_token 都合并得只剩一个 bytes，提前退出
                    break
                merge_candidate = sorted_pairs[-1][1]

                pre_tokens, frequency = self._merge(pre_tokens, frequency, sorted_pairs, inverted, merge_candidate)
                self.vocab[len(self.vocab)] = merge_candidate[0] + merge_candidate[1]
                self.merges.append(merge_candidate)
                pbar.update(1)

    def pre_tokenize(self) -> None:
        self.pre_tokens = self._pre_tokenize_parallel()

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            path = os.path.join(self.cache_dir, "pre_tokens.pkl")
            with open(path, "wb") as file:
                pkl.dump(
                    {
                        "vocab_size": self.vocab_size,
                        "special_tokens": self.special_tokens,
                        "pre_tokens": self.pre_tokens,
                    },
                    file,
                )

    def merge(self) -> None:
        if self.pre_tokens is None:
            if self.cache_dir is not None:
                path = os.path.join(self.cache_dir, "pre_tokens.pkl")
                with open(path, "rb") as f:
                    data = pkl.load(f)
                assert data["vocab_size"] == self.vocab_size and data["special_tokens"] == self.special_tokens
                self.pre_tokens = data["pre_tokens"]
            else:
                raise FileNotFoundError("cache dir is None")
        assert self.pre_tokens is not None
        self._bpe_merge(self.pre_tokens)

    def dump(self) -> None:
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(os.path.join(self.cache_dir, 'vocab.pkl'), 'wb') as f:
                pkl.dump(self.vocab, f)
            with open(os.path.join(self.cache_dir, 'merges.pkl'), 'wb') as f:
                pkl.dump(self.merges, f)
        else:
             raise FileNotFoundError("cache dir is None")


def build_tokenizer(config: DatasetConfig):
    tokenizer_builder = TokenizerBuilder(
        config.train_file, config.vocab_size, config.special_tokens, config.cache_dir, config.num_iteration, config.num_processes
    )
    tokenizer_builder.pre_tokenize()
    tokenizer_builder.merge()
    tokenizer_builder.dump()


if __name__ == '__main__':
    build_tokenizer(TinyStoryConfig)
    build_tokenizer(OWTConfig)
