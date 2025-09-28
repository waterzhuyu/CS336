import time
import os
import numpy as np

from itertools import islice

from cs336_basics.train_bpe import optimized_train_bpe_parallel, optimized_train_bpe_heap_parallel
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.data import build_sharded_dataset_parallel

if __name__ == "__main__":
    # train_bpe_tinystories
    # vocab_size 10,000
    # input_path = f"/workspace/guozuyu/lmfs/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # input_path = f"/workspace/guozuyu/lmfs/assignment1-basics/data/try.txt"
    input_path = f"/workspace/guozuyu/lmfs/assignment1-basics/data/owt_train.txt"
    # start_time = time.time()
    # vocab, merges = optimized_train_bpe_heap_parallel(input_path, vocab_size=32000, special_tokens=["<|endoftext|>"], num_processes=32)
    # print("vocab: ", vocab)
    # print("merges: ", merges)
    # end_time = time.time()
    # bpe_tokenizer = Tokenizer(vocab, merges)

    tokenizer_output_path = f"/workspace/guozuyu/lmfs/assignment1-basics/results/owt-train"
    # tokenizer_output_path = f"/workspace/guozuyu/lmfs/assignment1-basics/results/tinystories-train"
    # vocab_path = os.path.join(tokenizer_output_path, "vocab.json")
    # merges_path = os.path.join(tokenizer_output_path, "merges.json")

    # bpe_tokenizer.to_files(tokenizer_output_path)

    # print("Train time: ", end_time - start_time)

    # tokenizer_output_path = f"/workspace/guozuyu/lmfs/assignment1-basics/results/tinystories"
    vocab_path = os.path.join(tokenizer_output_path, "vocab.json")
    merges_path = os.path.join(tokenizer_output_path, "merges.json")
    bpe_tokenizer = Tokenizer.from_files(vocab_path, merges_path)

    # encoded_output = f"/workspace/guozuyu/lmfs/assignment1-basics/data/TinyStoriesV2-GPT4-train.npy"
    encoded_output = f"/workspace/guozuyu/lmfs/assignment1-basics/data/owt-train"
    start_time = time.time()
    idx_path = build_sharded_dataset_parallel(input_path, encoded_output, bpe_tokenizer.encode_shard)
    end_time = time.time()
    print("encode time: ", end_time - start_time)
    # with open(input_path, encoding="utf-8") as f, open(encoded_output, mode="ab") as out_f:
    #     encode_start_time = time.time()
    #     # f is a iterable, read it line-by-line
    #     ids_iter = bpe_tokenizer.encode_iterable(f, batch_size=2048, num_workers=32)

    #     while True:
    #         chunk = islice(ids_iter, 100000)
    #         ids = np.fromiter(chunk, dtype=np.uint16)
    #         if len(ids) == 0:
    #             break

    #         encoded_end_time = time.time()
    #         # print(f"Encode time of {len(ids)} tokens: {encoded_end_time - encode_start_time}")       
    #         ids.tofile(out_f)
    #     encoded_end_time = time.time()
    #     print(f"Encoded time: {encoded_end_time - encode_start_time}")


        # while (ids := np.fromiter(ids_iter, dtype=np.uint16, count=10000)).size > 0:
        #     encode_end_time = time.time()
        #     print("Encode time of 10000 tokens: ", encode_end_time - encode_start_time)

        #     ids.tofile(out_f)

    # encoded_output = f"/workspace/guozuyu/lmfs/assignment1-basics/data/owt.npy"

    # train_bpe_expts_owt
    # vocab_size 32,000