import time
import os
import numpy as np

from itertools import islice

import torch
from tqdm import tqdm

from cs336_basics.train_bpe import optimized_train_bpe_parallel, optimized_train_bpe_heap_parallel
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.data import build_sharded_dataset_parallel, get_batch


def main():
    input_path = f"/public/home/wangfei/user_home/gzy/proj/CS336/data/owt_train.txt"
    start_time = time.time()
    vocab, merges = optimized_train_bpe_heap_parallel(input_path, vocab_size=32000, special_tokens=["<|endoftext|>"], num_processes=16)
    # print("vocab: ", vocab)
    # print("merges: ", merges)
    end_time = time.time()
    bpe_tokenizer = Tokenizer(vocab, merges)

    tokenizer_output_path = f"results/tokenizer/owt-train-32k"
    bpe_tokenizer.to_files(tokenizer_output_path)

    print(f"It takes {end_time - start_time}s to train tokenizer on tinystories.")

# def encode(tokenizer_dir, data_path, output_dir):
#     vocab_filepath = f"{tokenizer_dir}/vocab.json"
#     merges_filepath = f"{tokenizer_dir}/merges.json"
#     tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath)
#     with open(data_path, "r") as f:
#         ids_iter = tokenizer.encode_iterable(f)
    
#         output_path = f"{output_dir}/owt_valid.npy"
#         np.save(output_path, np.fromiter(ids_iter, dtype=np.uint16))


def encode(tokenizer_dir, data_path, output_dir, total_lines):
    vocab_filepath = f"{tokenizer_dir}/vocab.json"
    merges_filepath = f"{tokenizer_dir}/merges.json"
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath)
    with open(data_path, "r") as f:
        ids_iter = tokenizer.encode_iterable(f, total_lines)
    
        # output_path = f"{output_dir}/TinyStoriesV2-GPT4-valid.npy"
        output_path = f"{output_dir}/tinystories_sample.npy"
        # output_path = f"{output_dir}/owt_valid.npy"
        np.save(output_path, np.fromiter(ids_iter, dtype=np.uint16))

if __name__ == "__main__":
    # tokenizer_dir = "results/tokenizer/owt-train-32k"
    tokenizer_dir = "results/tokenizer/tinystories-train-10k"
    # data_path = "data/owt_valid.txt"
    # data_path = "data/TinyStoriesV2-GPT4-valid.txt"
    data_path = "tests/fixtures/tinystories_sample.txt"
    output_dir = "results/npy"

    with open(data_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    t0 = time.time()
    encode(tokenizer_dir, data_path, output_dir, total_lines)
    t1 = time.time()
    t = t1 - t0
    throughput = os.path.getsize(data_path) / (1024 * 1024) / t
    print(f"encode cost time: {t}s, throughput approx {throughput} MB/s.")
