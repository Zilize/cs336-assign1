import os
import time
import regex as re
import pickle as pkl
from tqdm import tqdm
from typing import BinaryIO
from multiprocessing import Pool


class Tokenizer:
    def __init__(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        dataset_name: str,
        num_iterations: int = 10,
        num_processes: int = 20,
    ):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.dataset_name = dataset_name

        self.num_iterations = num_iterations
        self.num_processes = num_processes

        self.vocab: dict[bytes, int] = self._init_vocab()
        assert vocab_size >= len(self.vocab)
        self.merges: list[tuple[bytes, bytes]] = list()
        self._pre_tokens: dict[tuple[bytes, ...], int] | None = None

    def _init_vocab(self):
        vocabulary = {bytes([i]): i for i in range(256)}
        for special_token in self.special_tokens:
            vocabulary[special_token.encode('utf-8')] = len(vocabulary)
        return vocabulary

    @staticmethod
    def _find_chunk_boundaries(
            file: BinaryIO,
            start: int,
            end: int,
            desired_num_chunks: int,
            split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        chunk_size = (end - start) // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size + start for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = end

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = end
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    @staticmethod
    def _pre_tokenize(args: tuple[str, tuple[int, int], list[str]]) -> dict[tuple[bytes, ...], int]:
        input_path, scope, special_tokens = args
        with open(input_path, "rb") as f:
            start, end = scope
            f.seek(start)
            input_text = f.read(end - start).decode("utf-8", errors="ignore")

        split_pattern = '(' + '|'.join(re.escape(tok) for tok in special_tokens) + ')'
        split_text_list = re.split(split_pattern, input_text)

        pre_tokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokenization_pattern = re.compile(pre_tokenization_pattern)

        pre_token_list = list()
        for split_text in split_text_list:
            if split_text in special_tokens:
                continue
            split_text_pre_token_list = re.findall(pre_tokenization_pattern, split_text)
            pre_token_list.extend(split_text_pre_token_list)

        pre_token_list = [pre_token.encode('utf-8') for pre_token in pre_token_list]
        pre_tokens: dict[tuple[bytes, ...], int] = dict()
        for pre_token in pre_token_list:
            if len(pre_token) <= 1:
                continue
            pre_token_tuple = tuple(pre_token[i:i+1] for i in range(len(pre_token)))
            pre_tokens[pre_token_tuple] = pre_tokens.get(pre_token_tuple, 0) + 1
        # 可以返回预分词列表，但是以频率表的方式更节省空间
        return pre_tokens

    @staticmethod
    def _merge(
            pre_tokens: dict[tuple[bytes, ...], int],
            frequency: dict[tuple[bytes, bytes], int],
            inverted: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
            merge_candidate: tuple[bytes, bytes]
    ):
        queue = list(inverted[merge_candidate])
        for pre_token in queue:
            new_pre_token_list: list[bytes] = list()

            i = 0
            subtract_index_pair_set = set()  # index in pre_token
            add_index_set: set[int] = set()  # index in new_pre_token_list

            is_merged = False
            while i < len(pre_token):
                if i < len(pre_token) - 1 and (pre_token[i], pre_token[i + 1]) == merge_candidate:
                    is_merged = True
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

            assert is_merged == True
            if is_merged:
                new_pre_token = tuple(new_pre_token_list)

                # 1. 增量更新 frequency
                # 合并后消失的共现组，从频率表中减去；
                for subtract_index_pair in subtract_index_pair_set:
                    subtract_item = (pre_token[subtract_index_pair[0]], pre_token[subtract_index_pair[1]])
                    frequency[subtract_item] -= pre_tokens[pre_token]
                    assert frequency[subtract_item] >= 0
                    if frequency[subtract_item] == 0:
                        del frequency[subtract_item]

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
                    frequency[add_item] = frequency.get(add_item, 0) + pre_tokens[pre_token]

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

        del frequency[merge_candidate]
        del inverted[merge_candidate]
        return pre_tokens, frequency

    def _pre_tokenize_corpus(self) -> dict[tuple[bytes, ...], int]:
        boundaries_list: list[list[tuple[int, int]]] = list()
        with open(self.input_path, "rb") as file:
            # Get total file size in bytes
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            major_boundaries = self._find_chunk_boundaries(file, 0, file_size, self.num_iterations, b"<|endoftext|>")
            for start, end in zip(major_boundaries[:-1], major_boundaries[1:]):
                boundaries = list()
                minor_boundaries = self._find_chunk_boundaries(file, start, end, self.num_processes, b"<|endoftext|>")
                for minor_start, minor_end in zip(minor_boundaries[:-1], minor_boundaries[1:]):
                    boundaries.append((minor_start, minor_end))
                boundaries_list.append(boundaries)

        pre_tokens = dict()
        for boundaries in tqdm(boundaries_list):
            args = list()
            for boundary in boundaries:
                args.append((
                    self.input_path,
                    boundary,
                    self.special_tokens
                ))

            with Pool(processes=self.num_processes) as pool:
                results = pool.map(self._pre_tokenize, args)
                for result in results:
                    for key in result:
                        pre_tokens[key] = pre_tokens.get(key, 0) + result[key]
        return pre_tokens

    def _train_bpe_merges(self, pre_tokens: dict[tuple[bytes, ...], int]) -> None:
        # 要增量式维护两个数据结构：
        # frequency: pair的频率统计表，用于找到频率最大的pair；
        #            其实可以用优先队列来做，排序同时考虑频次和key的字典序，不过这里不是瓶颈，所以先不考虑了
        # inverted: pair的倒排索引，用于快速找到合并所影响到的pre_token
        frequency: dict[tuple[bytes, bytes], int] = dict()
        inverted: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = dict()
        for pre_token in tqdm(pre_tokens):
            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i + 1])
                frequency[pair] = frequency.get(pair, 0) + pre_tokens[pre_token]
                inverted.setdefault(pair, set()).add(pre_token)
                # pair_dict = inverted.setdefault(pair, {})
                # pair_dict[pre_token] = pair_dict.get(pre_token, 0) + 1

        with tqdm(total=self.vocab_size, initial=len(self.vocab), desc="BPE training") as pbar:
            while len(self.vocab) < self.vocab_size:
                if len(frequency) <= 0:
                    # 对于小数据集，可能达成每个 pre_token 都合并得只剩一个 bytes，提前退出
                    break
                max_frequency = max(frequency.values())
                candidates = list()
                for pair in frequency:
                    if frequency[pair] == max_frequency:
                        candidates.append(pair)
                merge_candidate: tuple[bytes, bytes] = max(candidates)

                pre_tokens, frequency = self._merge(pre_tokens, frequency, inverted, merge_candidate)
                self.vocab[merge_candidate[0] + merge_candidate[1]] = len(self.vocab)
                self.merges.append(merge_candidate)
                pbar.update(1)

    def _default_pre_token_cache_path(self) -> str:
        return os.path.join("../cache", self.dataset_name, "pre_tokens.pkl")

    def _default_tokenizer_result_dir(self) -> str:
        return os.path.join("data", self.dataset_name)

    def pre_tokenize(self, use_cache: bool = True) -> None:
        if use_cache:
            path = self._default_pre_token_cache_path()
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            pre_tokens = self._pre_tokenize_corpus()
            with open(path, "wb") as f:
                pkl.dump(
                    {
                        "vocab_size": self.vocab_size,
                        "special_tokens": self.special_tokens,
                        "pre_tokens": pre_tokens,
                    },
                    f,
                )
            self._pre_tokens = None
        else:
            self._pre_tokens = self._pre_tokenize_corpus()

    def bpe_merge(self, use_cache: bool = True) -> None:
        if use_cache:
            path = self._default_pre_token_cache_path()
            with open(path, "rb") as f:
                data = pkl.load(f)
            assert data["vocab_size"] == self.vocab_size and data["special_tokens"] == self.special_tokens
            pre_tokens = data["pre_tokens"]
        else:
            assert self._pre_tokens is not None
            pre_tokens = self._pre_tokens
            self._pre_tokens = None

        # todo: for debug
        # 根据错误点构建了一个能偶现问题的小 pre_tokens 数据集
        # inverted: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = dict()
        # for pre_token in tqdm(pre_tokens):
        #     for i in range(len(pre_token) - 1):
        #         pair = (pre_token[i], pre_token[i + 1])
        #         inverted.setdefault(pair, set()).add(pre_token)
        #
        # target_pair = (b'w', b'w')
        # target_pre_tokens = dict()
        # extend_pairs = set()
        # for pre_token_key in inverted[target_pair]:
        #     target_pre_tokens[pre_token_key]  = pre_tokens[pre_token_key]
        #     for i in range(len(pre_token_key) - 1):
        #         pair = (pre_token_key[i], pre_token_key[i + 1])
        #         extend_pairs.add(pair)
        # for extend_pair in extend_pairs:
        #     for pre_token_key in inverted[extend_pair]:
        #         target_pre_tokens[pre_token_key] = pre_tokens[pre_token_key]

        self._train_bpe_merges(pre_tokens)

    def build(self) -> None:
        self.pre_tokenize(use_cache=False)
        self.bpe_merge(use_cache=False)

    def dump(self) -> None:
        save_dir = self._default_tokenizer_result_dir()
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(self.vocab, f)
        with open(os.path.join(save_dir, 'merges.pkl'), 'wb') as f:
            pkl.dump(self.merges, f)


def main():
    tokenizer = Tokenizer(
        '../data/owt_train.txt', 32000, ['<|endoftext|>'], 'owt', num_iterations=10
    )
    # tokenizer.pre_tokenize()
    tokenizer.bpe_merge()
    tokenizer.dump()


if __name__ == '__main__':
    main()