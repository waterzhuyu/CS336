from __future__ import annotations
import os
import json
import base64
import heapq
import regex as re
from tqdm import tqdm

import numpy as np

from typing import Iterable, Iterator
from itertools import islice
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Constructor of `Tokenizer`.
        We can register special tokens in this contructor.
        
        Args: 
            vocab (dict[int, bytes])
            merges (list[tuple[bytes, bytes]])
            special_tokens (list[str]), special token could already exist in the vocab, so only append those haven't exist.

        """
        self.vocab = vocab
        self.merges = merges
        # maintain the `special_tokens` in the `Tokenizer` instance, in this variable, 
        # special_token is `str`, attention to the convertion between `bytes`
        special_tokens = special_tokens if special_tokens is not None else ["<|endoftext|>"]

        original_vocab_size = len(vocab)
        if special_tokens is not None:
            bytes_tokens = [token.encode("utf-8") for token in special_tokens if token.encode("utf-8") not in vocab.values()]
            self.vocab |= {
                ids + original_vocab_size: token 
                for ids, token in enumerate(bytes_tokens)
            }
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        self.special_tokens = {token: self.reverse_vocab[token.encode("utf-8")] for token in special_tokens}

        self.merge_priority = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Constructs and return a `Tokenizer` from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens.
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab = {int(k): base64.b64decode(v) for k, v in vocab.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                a, b = json.loads(line)
                merges.append((base64.b64decode(a), base64.b64decode(b)))

        return cls(vocab, merges, special_tokens)

    def to_files(self, output_path: str):
        """
        Serialize vocab and merges to disk in the given file path
        Using Base64 encoding, not utf-8 in case of invalid unicode
        """
        os.makedirs(output_path, exist_ok=True)

        vocab_filepath = os.path.join(output_path, "vocab.json")
        merges_filepath = os.path.join(output_path, "merges.json")

        with open(vocab_filepath, "w", encoding="utf-8") as f:
            vocab = {k: base64.b64encode(v).decode("utf-8") for k, v in self.vocab.items()}
            # When json.dump, automatically convert key from int to str, 'cause key must be str
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        with open(merges_filepath, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(json.dumps([base64.b64encode(a).decode("utf-8"), base64.b64encode(b).decode("utf-8")], ensure_ascii=False) + "\n")
    
    @staticmethod
    def pre_tokenization(text: str) -> list[tuple[bytes, ...]]:
        """
        Pre-Tokenization by regex to shorten processing time to get byte pairs, and mitigate the impact of punctuation.

        Args:
            text (str): corpus

        Returns:
            pre_tokens (dict[tuple[bytes, ...], int]): mapping from pre_tokens to its occurrence counts
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens = []
        for match in re.finditer(PAT, text):
            string_token = match.group() # match is Match object, not str
            byte_token = tuple(bytes([b]) for b in string_token.encode("utf-8"))
            pre_tokens.append(byte_token)
        
        return pre_tokens

    @staticmethod
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

    @staticmethod
    def merge_at_idx(token: tuple[bytes, ...], idx: int) -> tuple[bytes, ...]:
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
            if i == idx:
                merged_tokens.append(token[i] + token[i+1])
                i += 2
            else:
                merged_tokens.append(token[i])
                i += 1

        return tuple(merged_tokens)

    def split_special_tokens(self, text: str, special_tokens: list[str]) -> list[str]:
        # in case of overlapping special tokens
        PAT = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))
        chunks = re.split(f"({PAT})", text)
        return chunks
    
    def _encode_chunk(self, text_bytes: tuple[bytes, ...]) -> list[int]:
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        # ids = list(text_bytes)
        while len(text_bytes) >= 2:
            stats = defaultdict(int)
            for window in zip(text_bytes, text_bytes[1:]):
                stats[window] += 1
            
            # find the pair with the lowest merge index
            pair = min(stats, key=lambda p: self.merge_priority.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            # idx = self.reverse_vocab[pair]
            text_bytes = self.merge_byte_pair(text_bytes, pair)
            # ids = list(text_bytes)
            # ids = merge(ids, pair, idx)
        ids = [self.reverse_vocab[tok] for tok in text_bytes]
        return ids
    
    class _Node:
        """
        Dual-Linked-List Node, maintaining byte in a word, for supporting in-place mutating, i.e. merging.
        """
        __slots__ = ['value', 'prev', 'next', 'index', 'deleted']
        
        def __init__(self, value: bytes, index: int):
            self.value: bytes = value # byte to merge, (a half of pair)
            self.index: int = index # cause' node will be maintained in a list of node, `nodes`, using index to fast access
            self.prev: Tokenizer._Node | None = None
            self.next: Tokenizer._Node | None = None
            # cause' node will be maintrained in a list, `nodes`, so will not automatically collected by garbage collection
            self.deleted: bool = False # denote node is merged, is out of date
    
    class HeapItem:
        """Maintain byte-pairs by Min-Heap, where priority is rank of merging."""
        def __init__(self, priority, index, pair):
            self.priority = priority # priority of merges
            self.index = index # left of pair, index in `nodes`
            self.pair = pair
        
        def __lt__(self, other: Tokenizer.HeapItem):
            if self.priority != other.priority:
                return self.priority < other.priority
            if self.pair != other.pair:
                return self.pair < other.pair
            return self.index < other.index

    def _encode_chunk_optimized(self, text_bytes: tuple[bytes, ...]) -> list[int]:
        """
        maintain byte pair by heap and update merged byte pair to a token in-place by dual-linked-list.
        """
        if not text_bytes:
            return []

        # 1: Initialize the dual linked list
        nodes: list[Tokenizer._Node] = [Tokenizer._Node(value=b, index=i) for i, b in enumerate(text_bytes)]
        if len(nodes) > 1:
            for i in range(len(nodes) - 1):
                nodes[i].next = nodes[i+1]
                nodes[i+1].prev = nodes[i]

        # 2: Initialize the heap for byte pair and its priority
        pq: list[Tokenizer.HeapItem] = []

        for i in range(len(nodes) - 1):
            left, right = nodes[i], nodes[i+1]
            pair = (left.value, right.value)
            
            priority = self.merge_priority.get(pair, float("inf"))
            
            if priority != float("inf"):
                heapq.heappush(pq, Tokenizer.HeapItem(priority, i, pair))

        # 3: Merge a pair
        while pq:
            candidate_item = heapq.heappop(pq)
            
            left = nodes[candidate_item.index]
            
            # check effectiveness
            if left.deleted:
                continue
            right = left.next
            if right is None or right.deleted:
                continue
            if (left.value, right.value) != candidate_item.pair:
                continue

            left.deleted = True
            right.deleted = True
            
            # merge
            new_val = candidate_item.pair[0] + candidate_item.pair[1]

            # create a new node in dual-linked-list
            new_node = Tokenizer._Node(value=new_val, index=candidate_item.index)
            nodes[candidate_item.index] = new_node  # 
            
            new_node.prev = left.prev
            new_node.next = right.next
            if new_node.prev:
                new_node.prev.next = new_node
            if new_node.next:
                new_node.next.prev = new_node

            # --- add new byte pair to heap ---
            
            # left-side
            if new_node.prev:
                prev_node = new_node.prev
                left_pair = (prev_node.value, new_node.value)
                left_priority = self.merge_priority.get(left_pair, float("inf"))
                if left_priority != float("inf"):
                    heapq.heappush(pq, Tokenizer.HeapItem(left_priority, prev_node.index, left_pair))
            
            # right-side
            if new_node.next:
                next_node = new_node.next
                right_pair = (new_node.value, next_node.value)
                right_priority = self.merge_priority.get(right_pair, float("inf"))
                if right_priority != float("inf"):
                    heapq.heappush(pq, Tokenizer.HeapItem(right_priority, new_node.index, right_pair))

        # 5. convert dual-linked-list back to list, collect all bytes
        head = None
        for node in nodes:
            if not node.deleted and node.prev is None:
                head = node
                break
        if head is None:
            for node in nodes:
                if not node.deleted:
                    head = node
                    break
            if head is None:
                 return []
                 
        final_bytes: list[bytes] = []
        curr = head
        while curr:
            final_bytes.append(curr.value)
            curr = curr.next

        ids = [self.reverse_vocab[tok] for tok in final_bytes]
        return ids
    
    def encode_ordinary(self, text: str):
        """Encoding that ignore any special tokens. """
        # split text into chunks of text by categories defined in regex pattern, aka pre-tokenization
        text_chunks = Tokenizer.pre_tokenization(text)
        ids = []
        for chunk in text_chunks:
            chunk_ids = self._encode_chunk_optimized(chunk)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special="all"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_chunks = self.split_special_tokens(text, list(self.special_tokens.keys()))
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids

    def deoptimized_encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.

        Keys: 
            - To correctly tokenize the special tokens, can't do pre-tokenization on them 'cause it can corrupt their structure.
        So implement the `split_special_token` method.
            - `merges` actually have a order in the process of tokenizer's training. So we should merge byte pair in this order
        but not the order of iterating the pre_tokens. 

        Args:
            text (str): 

        Returns:
            target_ids (list[int]):
        """
        chunks = self.split_special_tokens(text, list(self.special_tokens.keys()))

        pre_tokens = []
        for chunk in chunks:
            # don't pre-tokenization the special tokens
            if chunk in self.special_tokens:
                pre_tokens.extend([chunk])
            else:
                pre_tokens.extend(Tokenizer.pre_tokenization(chunk))

        # merge byte pair by the order of `merges`
        for merge in self.merges:
            tokenized = [] # Every loop merge 1 byte pair, so `tokenized` is a intermediate variable after merge this pair
            for token in pre_tokens:
                if token in self.special_tokens:
                    # Don't tokenize special tokens
                    tokenized.append(token)
                    continue
                
                idx = 0
                while idx < len(token) - 1:
                    if (token[idx], token[idx+1]) == merge:
                        # don't increment idx because it could merge another time in this idx
                        token = Tokenizer.merge_byte_pair(token, merge)
                    else:
                        idx += 1
                tokenized.append(token)
            pre_tokens = tokenized
        
        # map the tokens in vocab to its id
        target_ids = []
        for token in pre_tokens:
            if token in self.special_tokens:
                target_ids.append(self.reverse_vocab[token.encode("utf-8")])
            else:
                target_ids.extend([self.reverse_vocab[t] for t in token])
        
        return target_ids
    
    def encode_heap(self, text: str):
        """Maintain a priority-heap."""
        chunks = self.split_special_tokens(text, list(self.special_tokens.keys()))

        pre_tokens = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                pre_tokens.extend([chunk])
            else:
                pre_tokens.extend(Tokenizer.pre_tokenization(chunk))

        target_ids = []
        # Construct heap
        heap = []
        for token in pre_tokens: # mutate when iterating, token is immutable

            if token in self.special_tokens:
                target_ids.append(self.reverse_vocab[token.encode("utf-8")])
            else:
                for i in range(len(token) - 1):
                    pair = (token[i], token[i+1])
                    if pair in self.merge_priority:
                        heapq.heappush(heap, (self.merge_priority[pair], i, pair))

                while heap:
                    if len(token) == 1:
                        # only  1 token now, don't need to iterate heap, all its content is outdated.
                        break

                    _, i, pair = heapq.heappop(heap)
                    # lazy deletion: detect this heap pair if outdated
                    # 'H' 'e' 'l' 'l' 'o', if merge 'H' 'e' first time, then (3, ('l', 'o')) need to update idx
                    if i >= len(token) - 1:
                        continue
                    if (token[i], token[i+1]) != pair:
                        continue

                    token = Tokenizer.merge_at_idx(token, i)

                    if i > 0:
                        # don't need to update its all left pair
                        left_pair = (token[i-1], token[i])
                        if left_pair in self.merge_priority:
                            heapq.heappush(heap, (self.merge_priority[left_pair], i - 1, left_pair))

                    while i < len(token) - 1:
                        # need to update all its right pair, because need to update its index
                        right_pair = (token[i], token[i+1])
                        if right_pair in self.merge_priority:
                            heapq.heappush(heap, (self.merge_priority[right_pair], i, right_pair))
                        i += 1

                target_ids.extend([self.reverse_vocab[t] for t in token])

        return target_ids

    def encode_iterable(self, iterable: Iterable[str], total: int | None = None) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into
        memory.

        Using ThreadPoolExecutor to parallelize the encoding of one batch.
        """
        for chunk in tqdm(iterable, total=total, desc="Tokenizing", unit="lines"):
            yield from self.encode(chunk)

    def encode_shard(self, shard_id, lines, out_dir, batch_size, num_workers):
        """encode batch and write to numpy file"""
        buffer = []
        shard_path = os.path.join(out_dir, f"shard_{shard_id:05d}.npy")

        with ThreadPoolExecutor(num_workers) as executor:
            for batch in batched(lines, batch_size):
                results = list(executor.map(self.encode, batch))
                buffer.extend(results)

        np.save(shard_path, np.array(buffer, dtype=object))
        print(f"[Shard {shard_id}] Saved {len(buffer)} samples -> {shard_path}")
        
        return shard_path, len(buffer)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = [self.vocab[i] for i in ids]
        return b''.join(tokens).decode("utf-8", errors="replace")

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 2) → AB CD EF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch
