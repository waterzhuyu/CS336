import os
import random
import json
import torch
import numpy.typing as npt
import numpy as np

from concurrent.futures import ProcessPoolExecutor

def get_batch(data: npt.NDArray, batch_size: int, context_length: int, device: str):
    torch_device = torch.device(device)

    batch_list = []
    target_list = []
    possible_start_idx = len(data) - context_length - 1
    for _ in range(batch_size):
        start_idx = random.randint(0, possible_start_idx)
        batch = torch.tensor(data[start_idx:start_idx+context_length], device=torch_device)
        target = torch.tensor(data[start_idx+1:start_idx+context_length+1], device=torch_device)

        batch_list.append(batch)
        target_list.append(target)

    return torch.stack(batch_list, dim=0), torch.stack(target_list, dim=0)

class Dataset:
    def __init__(self, out_dir) -> None:
        self.out_dir = out_dir
        index_path = os.path.join(out_dir, "index.json")
        with open(index_path) as f:
            meta = json.load(f)
        self.shards = meta["shards"]
        self.lengths = meta["length"]
        self.offsets = np.cumsum([0] + self.lengths)
        self.total_len = self.offsets[-1]
        self._cache = {}

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_len:
            raise IndexError("Index out of range")
        shard_id = np.searchsorted(self.offsets, idx, side="right") - 1
        local_idx = idx - self.offsets[shard_id]
        if shard_id not in self._cache:
            shard_path = self.out_dir / self.shards[shard_id]
            self._cache[shard_id] = np.load(shard_path, allow_pickle=True)
        return self._cache[shard_id][local_idx]

def build_sharded_dataset_parallel(
    txt_path: str,
    out_dir: str,
    encode_shard,
    shard_size: int = 50000,
    batch_size: int = 256,
    num_workers_per_shard: int = 8,
    num_shard_procs: int = 64
):
    """
    构建大文本数据集分片：
    - 并行处理每个 shard
    - shard 内批量并行 encode
    - 生成 index.json，支持随机索引
    """
    os.makedirs(out_dir, exist_ok=True)

    # Step1: 读取文本文件，划分每个 shard 的行
    shards_lines = []
    current_shard = []
    shard_id = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            current_shard.append(line.strip())
            if len(current_shard) >= shard_size:
                shards_lines.append((shard_id, current_shard.copy()))
                current_shard.clear()
                shard_id += 1
        if current_shard:
            shards_lines.append((shard_id, current_shard.copy()))
            shard_id += 1

    print(f"Total shards: {len(shards_lines)}")

    # Step2: 使用进程池并行处理 shard
    index_meta = {"shards": [], "lengths": []}

    with ProcessPoolExecutor(max_workers=num_shard_procs) as executor:
        futures = []
        for shard_id, lines in shards_lines:
            futures.append(
                executor.submit(
                    encode_shard,
                    shard_id,
                    lines,
                    out_dir,
                    batch_size,
                    num_workers_per_shard,
                )
            )
        for fut in futures:
            shard_file, num_samples = fut.result()
            index_meta["shards"].append(shard_file)
            index_meta["lengths"].append(num_samples)

    # Step3: 保存索引
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index_meta, f, indent=2)

    print(f"✅ Dataset built. Total shards: {len(index_meta['shards'])}, total samples: {sum(index_meta['lengths'])}")
    return index_path