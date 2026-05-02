import regex as re

text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""


class Tokenizer:
    def __init__(self):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.vocab: dict[bytes, int] = self._init_vocab()
        self.merges: list[tuple[bytes, bytes]] = list()

    @staticmethod
    def _init_vocab():
        vocabulary = {bytes([i]): i for i in range(256)}
        vocabulary['<|endoftext|>'.encode('utf-8')] = 256
        return vocabulary

    @staticmethod
    def _pre_tokenize(input_text: str) -> dict[tuple[bytes, ...], int]:
        pre_token_list = input_text.split()
        pre_token_list = [pre_token.encode('utf-8') for pre_token in pre_token_list]
        pre_tokens: dict[tuple[bytes, ...], int] = dict()
        for pre_token in pre_token_list:
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
            new_pre_token = list()

            i = 0
            while i < len(pre_token):
                if i < len(pre_token) - 1 and (pre_token[i], pre_token[i + 1]) == merge_candidate:
                    merged = merge_candidate[0] + merge_candidate[1]
                    new_pre_token.append(merged)
                    # 发生合并，需要考虑对前后 pair 的影响
                    # (pre_token[i - 1], pre_token[i]) 的频次转变为 (pre_token[i - 1], merged) 的频次
                    # (pre_token[i + 1], pre_token[i + 2]) 的频次转变为 (merged, pre_token[i + 2]) 的频次
                    if i - 1 >= 0:
                        front_pair = (pre_token[i - 1], pre_token[i])
                        frequency[front_pair] -= pre_tokens[pre_token]
                        if frequency[front_pair] == 0:
                            del frequency[front_pair]
                        merged_front_pair = (pre_token[i - 1], merged)
                        frequency[merged_front_pair] = frequency.get(merged_front_pair, 0) + pre_tokens[pre_token]
                    if i + 2 <= len(pre_token) - 1:
                        back_pair = (pre_token[i + 1], pre_token[i + 2])
                        frequency[back_pair] -= pre_tokens[pre_token]
                        if frequency[back_pair] == 0:
                            del frequency[back_pair]
                        merged_back_pair = (merged, pre_token[i + 2])
                        frequency[merged_back_pair] = frequency.get(merged_back_pair, 0) + pre_tokens[pre_token]
                    i += 2
                else:
                    new_pre_token.append(pre_token[i])
                    i += 1

            new_pre_token = tuple(new_pre_token)
            new_pre_tokens[new_pre_token] = pre_tokens[pre_token]

        del frequency[merge_candidate]
        return new_pre_tokens, frequency


    def tokenize(self, input_text: str, vocab_size: int):
        assert vocab_size >= len(self.vocab)

        pre_tokens = self._pre_tokenize(input_text)
        frequency: dict[tuple[bytes, bytes], int] = dict()
        for pre_token in pre_tokens:
            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i + 1])
                frequency[pair] = frequency.get(pair, 0) + pre_tokens[pre_token]

        while len(self.vocab) < vocab_size:
            max_frequency = max(frequency.values())
            candidates = list()
            for pair in frequency:
                if frequency[pair] == max_frequency:
                    candidates.append(pair)
            merge_candidate: tuple[bytes, bytes] = max(candidates)

            pre_tokens, frequency = self._merge(pre_tokens, frequency, merge_candidate)
            self.vocab[merge_candidate[0] + merge_candidate[1]] = len(self.vocab)
            self.merges.append(merge_candidate)



def main():
    tokenizer = Tokenizer()
    tokenizer.tokenize(text, 1000)


if __name__ == '__main__':
    main()