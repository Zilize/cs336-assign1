import regex as re
from tqdm import tqdm

text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""


class Tokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        self.vocab: dict[bytes, int] = self._init_vocab()
        assert vocab_size >= len(self.vocab)
        self.merges: list[tuple[bytes, bytes]] = list()

        self.build()

    def _init_vocab(self):
        vocabulary = {bytes([i]): i for i in range(256)}
        for special_token in self.special_tokens:
            vocabulary[special_token.encode('utf-8')] = len(vocabulary)
        return vocabulary

    def _pre_tokenize(self, input_text: str) -> dict[tuple[bytes, ...], int]:
        split_pattern = '(' + '|'.join(re.escape(tok) for tok in self.special_tokens) + ')'
        split_text_list = re.split(split_pattern, input_text)

        pre_tokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokenization_pattern = re.compile(pre_tokenization_pattern)

        pre_token_list = list()
        for split_text in split_text_list:
            if split_text in self.special_tokens:
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

    def build(self):
        input_text = open(self.input_path, 'r', encoding='utf-8').read()
        pre_tokens = self._pre_tokenize(input_text)
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


def main():
    tokenizer = Tokenizer('/root/cs336-assign1/data/TinyStoriesV2-GPT4-valid.txt', 10000, ['<|endoftext|>'])
    pass


if __name__ == '__main__':
    main()