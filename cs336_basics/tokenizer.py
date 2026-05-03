import os
import regex as re
import pickle as pkl
from tqdm import tqdm
from typing import BinaryIO
from multiprocessing import Pool


class Tokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = 24

        self.vocab: dict[bytes, int] = self._init_vocab()
        assert vocab_size >= len(self.vocab)
        self.merges: list[tuple[bytes, bytes]] = list()

        self._build()

    def _init_vocab(self):
        vocabulary = {bytes([i]): i for i in range(256)}
        for special_token in self.special_tokens:
            vocabulary[special_token.encode('utf-8')] = len(vocabulary)
        return vocabulary

    @staticmethod
    def _find_chunk_boundaries(
            file: BinaryIO,
            desired_num_chunks: int,
            split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
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
        for split_text in tqdm(split_text_list):
            if split_text in special_tokens:
                continue
            split_text_pre_token_list = re.findall(pre_tokenization_pattern, split_text)
            pre_token_list.extend(split_text_pre_token_list)

        pre_token_list = [pre_token.encode('utf-8') for pre_token in pre_token_list]
        pre_tokens: dict[tuple[bytes, ...], int] = dict()
        for pre_token in tqdm(pre_token_list):
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
            merge_candidate: tuple[bytes, bytes]
    ):
        new_pre_tokens = dict()
        for pre_token in pre_tokens:
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

            # 合并后消失的共现组，从频率表中减去
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

            new_pre_token = tuple(new_pre_token_list)
            new_pre_tokens[new_pre_token] = pre_tokens[pre_token]

        del frequency[merge_candidate]
        return new_pre_tokens, frequency

    def _build(self):
        args = list()
        with open(self.input_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                args.append((
                    self.input_path,
                    (start, end),
                    self.special_tokens
                ))

        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self._pre_tokenize, args)

        key_set = set()
        for result in results:
            key_set.update(list(result.keys()))
        pre_tokens = dict()
        for key in key_set:
            for result in results:
                if key in result:
                    pre_tokens[key] = pre_tokens.get(key, 0) + result[key]

        frequency: dict[tuple[bytes, bytes], int] = dict()
        for pre_token in pre_tokens:
            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i + 1])
                frequency[pair] = frequency.get(pair, 0) + pre_tokens[pre_token]

        with tqdm(total=self.vocab_size, initial=len(self.vocab), desc="BPE training") as pbar:
            while len(self.vocab) < self.vocab_size:
                max_frequency = max(frequency.values())
                candidates = list()
                for pair in frequency:
                    if frequency[pair] == max_frequency:
                        candidates.append(pair)
                merge_candidate: tuple[bytes, bytes] = max(candidates)

                pre_tokens, frequency = self._merge(pre_tokens, frequency, merge_candidate)
                self.vocab[merge_candidate[0] + merge_candidate[1]] = len(self.vocab)
                self.merges.append(merge_candidate)
                pbar.update(1)


def process_dataset(dataset_name, dataset_path, vocab_size):
    tokenizer = Tokenizer(dataset_path, vocab_size, ['<|endoftext|>'])
    save_dir = os.path.join(f'data/{dataset_name}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'vocab.pkl'), 'wb') as f:
        pkl.dump(tokenizer.vocab, f)
    with open(os.path.join(save_dir, 'merges.pkl'), 'wb') as f:
        pkl.dump(tokenizer.merges, f)


def main():
    process_dataset('tinystory', 'data/TinyStoriesV2-GPT4-train.txt', 10000)
    process_dataset('owt', 'data/owt_train.txt', 32000)


if __name__ == '__main__':
    main()