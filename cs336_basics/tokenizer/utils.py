import regex as re
from typing import BinaryIO


def find_chunk_boundaries(
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


def pre_tokenize_from_text(input_text: str, special_tokens: list[str] | None) -> list[tuple[bytes, ...]]:
    if special_tokens is not None:
        split_pattern = '(' + '|'.join(re.escape(tok) for tok in special_tokens) + ')'
        split_text_list = re.split(split_pattern, input_text)
    else:
        split_text_list = [input_text]

    pre_tokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenization_pattern = re.compile(pre_tokenization_pattern)

    pre_token_list = list()
    for split_text in split_text_list:
        if special_tokens is not None and split_text in special_tokens:
            continue
        split_text_pre_token_list = re.findall(pre_tokenization_pattern, split_text)
        pre_token_list.extend(split_text_pre_token_list)

    pre_token_list = [pre_token.encode('utf-8') for pre_token in pre_token_list]
    pre_tokens: list[tuple[bytes, ...]] = list()
    for pre_token in pre_token_list:
        pre_token_tuple = tuple(pre_token[i:i + 1] for i in range(len(pre_token)))
        pre_tokens.append(pre_token_tuple)
    # 可以返回预分词列表，但是以频率表的方式更节省空间
    return pre_tokens


def pre_tokenize_from_file(
        input_path: str,
        start: int,
        end: int,
        special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        input_text = f.read(end - start).decode("utf-8", errors="ignore")

    pre_token_list = pre_tokenize_from_text(input_text, special_tokens)
    pre_tokens: dict[tuple[bytes, ...], int] = dict()
    for pre_token in pre_token_list:
        if len(pre_token) <= 1:
            continue
        pre_tokens[pre_token] = pre_tokens.get(pre_token, 0) + 1
    # 可以返回预分词列表，但是以频率表的方式更节省空间
    return pre_tokens
