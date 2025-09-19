import json

import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Constructor of `Tokenizer`.
        
        Args: 
            vocab (dict[int, bytes])
            merges (list[tuple[bytes, bytes]])
            special_tokens (list[str]), special token could already exist in the vocab, so only append those haven't exist.

        """
        self.vocab = vocab
        self.merges = merges
        # maintain the `special_tokens` in the `Tokenizer` instance, in this variable, 
        # special_token is `str`, attention to the convertion between `bytes`
        self.special_tokens = special_tokens if special_tokens is not None else ["<|endoftext|>"]

        original_vocab_size = len(vocab)
        if special_tokens is not None:
            bytes_tokens = [token.encode("utf-8") for token in special_tokens if token.encode("utf-8") not in vocab.values()]
            self.vocab |= {
                ids + original_vocab_size: token 
                for ids, token in enumerate(bytes_tokens)
            }
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Constructs and return a `Tokenizer` from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens.
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab = {k: v.encode("utf-8") for k, v in vocab.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            line = f.readline()
            a, b = json.loads(line)
            merges.append((a.encode("utf-8"), b.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def to_files(self, output_path: str):
        """
        Serialize vocab and merges to disk in the given file path
        """
        vocab_filepath = f"{output_path}/vocab.json"
        merges_filepath = f"{output_path}/merges.json"
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            vocab = {k: v.decode("utf-8", errors="replace") for k, v in self.vocab.items()}
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        with open(merges_filepath, "w", encoding="utf-8") as f:
            for a, b in merges:
                f.write(json.dumps([a.decode("utf-8"), b.decode("utf-8")], ensure_ascii=False) + "\n")
    
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

    def split_special_tokens(self, text, special_tokens):
        # in case of overlapping special tokens
        PAT = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))
        chunks = re.split(f"({PAT})", text)

        return chunks

    #FIXME: Following implementation is false, 'cause `merges` is actually ordered, we should 
    # merge byte pair by this order, not by the order of bytes in a pre_token
    # def encode(self, text: str) -> list[int]:
    #     """
    #     Encode an input text into a sequence of token IDs.

    #     Args:
    #         text (str): 

    #     Returns:
    #         target_ids (list[int]):
    #     """
    #     chunks = self.split_special_tokens(text, self.special_tokens)

    #     # pre_tokens = Tokenizer.pre_tokenization(text)
    #     pre_tokens = []
    #     for chunk in chunks:
    #         # don't pre-tokenization the special tokens
    #         if chunk in self.special_tokens:
    #             pre_tokens.extend([chunk])
    #         else:
    #             pre_tokens.extend(Tokenizer.pre_tokenization(chunk))

    #     target_ids = []
    #     for token in pre_tokens:
    #         # encode special tokens
    #         if token in self.special_tokens:
    #             target_ids.extend([self.reverse_vocab[token.encode("utf-8")]])
    #             continue

    #         idx, length = 0, len(token)
    #         # while idx < length - 1:
    #         #     if token[idx] + token[idx+1] in self.vocab.values():
    #         #         token = Tokenizer.merge_byte_pair(token, (token[idx], token[idx+1]))
    #         #         length -= 1
    #         #         if idx > 0:
    #         #             # backward 1 step to check if they can merge
    #         #             # idx -= 1
    #         #             idx = 0
    #         #     else:
    #         #         idx += 1
    #         # Appling the merge rules learned ```in order``` on the splits
    #         windows = list(zip(token, token[1:]))
    #         for merge in self.merges:
    #             if merge in windows:
    #                 token = Tokenizer.merge_byte_pair(token, merge)

    #         target_ids.extend([self.reverse_vocab[t] for t in token])
    #     return target_ids

    def encode(self, text: str) -> list[int]:
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
        chunks = self.split_special_tokens(text, self.special_tokens)

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
                
                #FIXME: The commentted implementation is buggy. 
                # When merge a byte pair in a pre_token, this pair could occur not single one time in the pre_token
                # So we should substract the counts of the occurrence of this pair, not only 1, or it could throw out-of-index. 
                # idx, length = 0, len(token)
                # while idx < length - 1:
                #     if (token[idx], token[idx+1]) == merge:
                #         token = Tokenizer.merge_byte_pair(token, merge) # merge all byte pairs in the token
                #         length -= 1 # length substract 1???
                #     else: 
                #         idx += 1
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

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        """
        for elem in iterable:
            yield from self.encode(elem)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = [self.vocab[i] for i in ids]
        return b''.join(tokens).decode("utf-8", errors="replace")


if __name__ == "__main__":
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at', 11: b'<|endoftext|>'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    special_tokens = ["<|endoftext|>"]
    PAT = "|".join(map(re.escape, special_tokens))
    chunks = re.split(f"({PAT})", "the cat ate <|endoftext|><|endoftext|>")
    tokenizer = Tokenizer(vocab, merges)
    ids = tokenizer.encode("the cat ate <|endoftext|><|endoftext|>")
    decoded_str = tokenizer.decode(ids)

    tokenizer.to_files("results")

    a = Tokenizer.from_files("results/vocab.json", "results/merges.json")

    print(a.vocab, a.merges)
