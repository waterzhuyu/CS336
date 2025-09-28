import os
import time
import multiprocessing
import heapq
import tqdm
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_chunk_worker(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        striped_chunk = strip_special_tokens(chunk, special_tokens)
        pre_tokens = Counter()
        for chunk in tqdm.tqdm(striped_chunk, desc=f"Processing sub-chunk of {start} to {end}"):
            pre_tokens.update(pre_tokenization(chunk))
        # pre_tokens = [pre_tokenization(docs) for docs in striped_chunk]
        # print(f"Complete pre_tokens before construct counter")
        # pre_tokens = dict(sum((Counter(d) for d in pre_tokens), Counter()))
        # print(f"Completed from {start} to {end}.")
    return pre_tokens

def train_bpe(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, trains a (byte-level) BPE tokenizer.

    Args: 
        input_path (str | os.PathLike): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer that defines the maximum final vocabulary size (including the
            initial byte vocabulary, vocabulary items produced from merging, and any special tokens)
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special tokens do not
            otherwise affect BPE training.

    Returns: 
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes).
        merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training. Each list item
            is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
            <token2>. The merges should be ordered by order of creation.
    """
    # init vocab
    vocab = {ids: bytes([ids]) for ids in range(256)} | \
        {256 + ids: token.encode("utf-8") for ids, token in enumerate(special_tokens)}
    curr_size = len(vocab)
    merges = []

    with open(input_path, "r", encoding="utf-8") as f:
        content: str = f.read()
        striped_content = strip_special_tokens(content, special_tokens)

        pre_tokens = [pre_tokenization(docs) for docs in striped_content]
        # combine pre_tokens from different docs, sum their counts
        pre_tokens = dict(sum((Counter(d) for d in pre_tokens), Counter()))

        while curr_size < vocab_size:
            pre_tokens, pair_to_merge = merge_once(pre_tokens)

            vocab[len(vocab)] = b''.join(pair_to_merge) # b''.join(pair_to_merge) to merge tuple[bytes, bytes] to a bytes
            merges.append(pair_to_merge)
            curr_size += 1

    return vocab, merges

def optimized_train_bpe(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, trains a (byte-level) BPE tokenizer.

    API change, from `merge_once` to `optimized_merge`, which update `pre_tokens`, `byte_pairs` and `byte_pair_positions` in place,
    and using incremental update.

    Args: 
        input_path (str | os.PathLike): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer that defines the maximum final vocabulary size (including the
            initial byte vocabulary, vocabulary items produced from merging, and any special tokens)
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special tokens do not
            otherwise affect BPE training.

    Returns: 
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes).
        merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training. Each list item
            is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
            <token2>. The merges should be ordered by order of creation.
    """
    # init vocab
    vocab = {ids: bytes([ids]) for ids in range(256)} | \
        {256 + ids: token.encode("utf-8") for ids, token in enumerate(special_tokens)}
    curr_size = len(vocab)
    merges = []

    with open(input_path, "r", encoding="utf-8") as f:
        content: str = f.read()
        striped_content = strip_special_tokens(content, special_tokens)

        pre_tokens = [pre_tokenization(docs) for docs in striped_content]
        # combine pre_tokens from different docs, sum their counts
        pre_tokens = dict(sum((Counter(d) for d in pre_tokens), Counter()))

        byte_pairs, byte_pair_positions = init_byte_pair(pre_tokens)
        while curr_size < vocab_size:
            pre_tokens, pair_to_merge, byte_pairs, byte_pair_positions = optimized_merge(pre_tokens, byte_pairs, byte_pair_positions)

            vocab[len(vocab)] = b''.join(pair_to_merge) # b''.join(pair_to_merge) to merge tuple[bytes, bytes] to a bytes
            merges.append(pair_to_merge)
            curr_size += 1

    return vocab, merges

def optimized_train_bpe_parallel(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str], 
    num_processes: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, trains a (byte-level) BPE tokenizer.

    Parallalize the process of pre-tokenzition.

    Args: 
        input_path (str | os.PathLike): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer that defines the maximum final vocabulary size (including the
            initial byte vocabulary, vocabulary items produced from merging, and any special tokens)
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special tokens do not
            otherwise affect BPE training.

    Returns: 
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes).
        merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training. Each list item
            is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
            <token2>. The merges should be ordered by order of creation.
    """
    # init vocab
    vocab = {ids: bytes([ids]) for ids in range(256)} | \
        {256 + ids: token.encode("utf-8") for ids, token in enumerate(special_tokens)}
    curr_size = len(vocab)
    merges = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_chunk_worker, tasks)
    
    pre_tokens = Counter()
    for r in results:
        pre_tokens.update(r)

    byte_pairs, byte_pair_positions = init_byte_pair(pre_tokens)
    while curr_size < vocab_size:
        pre_tokens, pair_to_merge, byte_pairs, byte_pair_positions = optimized_merge(pre_tokens, byte_pairs, byte_pair_positions)

        vocab[len(vocab)] = b''.join(pair_to_merge) # b''.join(pair_to_merge) to merge tuple[bytes, bytes] to a bytes
        merges.append(pair_to_merge)
        curr_size += 1

    return vocab, merges

def merge_once(pre_tokens: dict[tuple, int]) -> tuple[dict[tuple, int], tuple[bytes, bytes]]:
    """
    Compute byte pair from pre_tokens, and extract the most frequent one to merge, by lexicographical order.

    Args:
        pre_tokens (dict[tuple, int]): a mapping from pre_tokens to occurrence counts
    
    Returns:
        pre_tokens (dict[tuple, int]): mapping that after merge a byte pair
        pair_to_merge (tuple[bytes, bytes]): byte pair that need to merge this round
    """
    byte_pair = {}
    for token, freq in pre_tokens.items():
        windows = zip(token, token[1:])
        for window in windows:
            byte_pair[window] = byte_pair.get(window, 0) + freq

    max_val = max(byte_pair.values())
    pair_to_merge = max([k for k, v in byte_pair.items() if v == max_val])
    pre_tokens = {merge_byte_pair(k, pair_to_merge): v for k, v in pre_tokens.items()}

    return pre_tokens, pair_to_merge

def init_byte_pair(pre_tokens: dict[tuple, int]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], list]]:
    """
    Initialize the `byte_pairs` and `byte_pair_positions` from `pre_tokens`

    Args:
        pre_tokens: mapping from pre_tokens to its occurrence counts
    
    Returns:
        byte_pairs: byte pairs
        byte_pair_positions: mapping from byte pair to the pre-tokens which it occurs.
    """
    byte_pair = defaultdict(int)
    byte_pair_postions = defaultdict(list)

    for token, freq in pre_tokens.items():
        windows = zip(token, token[1:])
        for window in windows:
            byte_pair[window] += freq
            byte_pair_postions[window].append(token)
    
    return byte_pair, byte_pair_postions

class HeapItem:
    def __init__(self, frequency: int, byte_pair: tuple[bytes, bytes]) -> None:
        self.frequency = frequency
        self.byte_pair = byte_pair

    def __lt__(self, other: "HeapItem"):
        if self.frequency != other.frequency:
            return self.frequency > other.frequency
        return self.byte_pair > other.byte_pair

def init_byte_pair_heapq(pre_tokens: dict[tuple, int]) -> tuple[dict[tuple[bytes, bytes], int], list[HeapItem], dict[tuple[bytes, bytes], set]]:
    """
    Maintain byte-pair order by max-heap. Reduce max() operation to O(1).
    
    Maintain byte_pair positions (mapping from byte pair to its occurred word) in a dict[tuple, set], 
    reduce add/remove from this set to O(1)
    """
    byte_pair = defaultdict(int)
    byte_pair_positions = defaultdict(set)

    for token, freq in pre_tokens.items():
        windows = zip(token, token[1:])
        for window in windows:
            byte_pair[window] += freq
            byte_pair_positions[window].add(token)

    # Construct a max heap
    heap_byte_pair = [HeapItem(freq, item) for item, freq in byte_pair.items()]
    heapq.heapify(heap_byte_pair)
    
    return byte_pair, heap_byte_pair, byte_pair_positions

def optimized_merge_heapq(
        pre_tokens: dict[tuple, int], 
        byte_pairs: dict[tuple[bytes, bytes], int], 
        heap_byte_pairs: list[HeapItem],
        byte_pair_positions: dict[tuple[bytes, bytes], set]
        ) -> tuple[dict[tuple, int], None | tuple[bytes, bytes], dict[tuple[bytes, bytes], int], list[HeapItem], dict[tuple[bytes, bytes], set]]:
    """
    Optimized the merging process by incremental updating and in-place update.
    API change from merge_once, which not maintain the `byte_pairs` and `byte_pair_positions` and create from scratch every time.
    """
    # TODO: max need linear complexity.
    max_heap_item = None
    while heap_byte_pairs:
        candidate_item = heapq.heappop(heap_byte_pairs)
        # lazy deletion
        if candidate_item.byte_pair not in byte_pairs:
            # this item have merged
            continue
        if candidate_item.frequency != byte_pairs[candidate_item.byte_pair]:
            # this item don't have a sync frequency with byte pairs dictionary
            continue

        max_heap_item = candidate_item
        break
    
    if max_heap_item is None:
        return pre_tokens, None, byte_pairs, heap_byte_pairs, byte_pair_positions

    pair_to_merge = max_heap_item.byte_pair
    positions = byte_pair_positions[pair_to_merge]

    affected_pairs = set()

    for token in list(positions): # is case of mutating in iterating
        freq = pre_tokens.pop(token, 0)
        if freq == 0:
            continue

        merged_token = merge_byte_pair(token, pair_to_merge)
        # pre_tokens[merged_token] += freq
        pre_tokens[merged_token] = pre_tokens.get(merged_token, 0) + freq

        for w in zip(token, token[1:]):
            # Update `byte_pairs` and `byte_pair_positions` but not `heap_byte_pairs`, 'cause its lazy deletion
            byte_pairs[w] -= freq
            byte_pair_positions[w].discard(token)
            if byte_pairs[w] <= 0:
                del byte_pairs[w]
                if w in byte_pair_positions:
                    del byte_pair_positions[w]           
            
            affected_pairs.add(w)

        for w in zip(merged_token, merged_token[1:]):
            byte_pairs[w] += freq
            byte_pair_positions[w].add(merged_token)
            affected_pairs.add(w)
    
    for pair in affected_pairs:
        if pair in byte_pairs:
            heapq.heappush(heap_byte_pairs, HeapItem(byte_pairs[pair], pair))

    return pre_tokens, pair_to_merge, byte_pairs,heap_byte_pairs, byte_pair_positions

def optimized_train_bpe_heap_parallel(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str], 
    num_processes: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, trains a (byte-level) BPE tokenizer.

    Parallalize the process of pre-tokenzition.

    Args: 
        input_path (str | os.PathLike): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer that defines the maximum final vocabulary size (including the
            initial byte vocabulary, vocabulary items produced from merging, and any special tokens)
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special tokens do not
            otherwise affect BPE training.

    Returns: 
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes).
        merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training. Each list item
            is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
            <token2>. The merges should be ordered by order of creation.
    """
    start_time = time.time()
    # init vocab
    vocab = {ids: bytes([ids]) for ids in range(256)} | \
        {256 + ids: token.encode("utf-8") for ids, token in enumerate(special_tokens)}
    curr_size = len(vocab)
    merges = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    pre_tokens = Counter()
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.imap_unordered(process_chunk_worker, tasks)
        # for counter in tqdm.tqdm(results, total=len(tasks), desc="Processing pre-tokenization"):
        # pre_tokens.update(counter)
        pre_tokens = sum(results, Counter())

    pre_tokenization_end_time = time.time()
    print("Pre-tokenization time: ", pre_tokenization_end_time - start_time)

    byte_pairs, heap_byte_pairs, byte_pair_positions = init_byte_pair_heapq(pre_tokens)
    # while curr_size < vocab_size:
    for _ in tqdm.tqdm(range(vocab_size - curr_size)):
        pre_tokens, pair_to_merge, byte_pairs, heap_byte_pairs, byte_pair_positions = optimized_merge_heapq(pre_tokens, byte_pairs, heap_byte_pairs, byte_pair_positions)
        if pair_to_merge is not None:
            vocab[len(vocab)] = b''.join(pair_to_merge) # b''.join(pair_to_merge) to merge tuple[bytes, bytes] to a bytes
            merges.append(pair_to_merge)
            # curr_size += 1
    print("Merge time:", time.time() - pre_tokenization_end_time)
    return vocab, merges

def optimized_merge(
        pre_tokens: dict[tuple, int], 
        byte_pairs: dict[tuple[bytes, bytes], int], 
        byte_pair_positions: dict[tuple[bytes, bytes], list]
        ) -> tuple[dict[tuple, int], tuple[bytes, bytes], dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], list]]:
    """
    Optimized the merging process by incremental updating and in-place update.
    API change from merge_once, which not maintain the `byte_pairs` and `byte_pair_positions` and create from scratch every time.
    """
    # TODO: max need linear complexity.
    max_val = max(byte_pairs.values())
    pair_to_merge = max([k for k, v in byte_pairs.items() if v == max_val])
    positions = set(byte_pair_positions[pair_to_merge])

    for token in list(positions):
        freq = pre_tokens.pop(token, 0)
        if freq == 0:
            continue

        merged_token = merge_byte_pair(token, pair_to_merge)
        # pre_tokens[merged_token] += freq
        pre_tokens[merged_token] = pre_tokens.get(merged_token, 0) + freq

        for w in zip(token, token[1:]):
            byte_pairs[w] -= freq
            if byte_pairs[w] <= 0:
                del byte_pairs[w]
            byte_pair_positions[w].remove(token)
        
        for w in zip(merged_token, merged_token[1:]):
            byte_pairs[w] += freq
            byte_pair_positions[w].append(merged_token)
    
    return pre_tokens, pair_to_merge, byte_pairs, byte_pair_positions
            
def merge_byte_pair(token: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    Given the pair_to_merge, merge two neighbors if this pair is in token, otherwise stay same.

    Args:
        token (tuple[bytes, ...]): 
        pair (tuple[bytes, bytes]):
    
    Returns: 
        merged_tokens (tuple[bytes, ...]):
    """
    merged_tokens = []
    i = 0
    while i < len(token):
        if i < len(token) - 1 and (token[i], token[i+1]) == pair:
            merged_tokens.append(token[i] + token[i+1])
            i += 2
        else:
            merged_tokens.append(token[i])
            i += 1
    
    return tuple(merged_tokens)

def pre_tokenization(text: str) -> dict[tuple[bytes, ...], int]:
    """
    Pre-Tokenization by regex to shorten processing time to get byte pairs, and mitigate the impact of punctuation.

    Args:
        text (str): corpus

    Returns:
        pre_tokens (dict[tuple[bytes, ...], int]): mapping from pre_tokens to its occurrence counts
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tokens = {}
    for match in re.finditer(PAT, text):
        string_token = match.group() # match is Match object, not str
        byte_token = tuple(bytes([b]) for b in string_token.encode("utf-8"))
        pre_tokens[byte_token] = pre_tokens.get(byte_token, 0) + 1
    
    return pre_tokens

def strip_special_tokens(chunk: str, special_tokens: list[str]) -> list[str]:
    """
    Strip the special tokens like <|endoftext|>, in case of cross document merge in byte pair.

    Args:
        chunk (str): corpus or chunk of corpus
        special_tokens (list[str]): special tokens to strip

    Returns: 
        striped_chunk (list[str]): corpus that after stripping
    """
    PAT = "|".join(special_tokens)
    striped_chunk = re.split(PAT, chunk)

    return striped_chunk

def find_chunk_boundaries(
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

# if __name__ == "__main__":
#     input_path = "/workspace/guozuyu/lmfs/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"

#     start_time = time.time()
#     vocab, merges = optimized_train_bpe_heap_parallel(input_path=input_path, vocab_size=10000, special_tokens=["<|endoftext|>"])
#     end_time = time.time()
#     print(end_time - start_time)