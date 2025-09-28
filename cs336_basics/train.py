import numpy as np

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from .data import get_batch
from .models import TransformerLM
from .optimizer import AdamW
from .nn_utils import cross_entropy, gradient_clipping
from .serialization import save_checkpoint, load_checkpoint

def train(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        rope_theta,
        lr,
        betas,
        eps,
        weight_decay,
        data_path,
        batch_size,
        steps,
        max_norm,
        save_path,
    ):

    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta)
    optimizer = AdamW(model.parameters(), lr, betas, eps, weight_decay)

    datas = np.memmap(data_path, dtype=str)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i in range(steps):
        data, target = get_batch(datas, batch_size, context_length, device)

        optimizer.zero_grad()
        logits = model(data)
        loss = cross_entropy(logits, target)
        print(f"At step {i}, loss is ", loss.cpu().item())
        loss.backward()
        gradient_clipping(model.parameters(), max_norm)

        optimizer.step()

        if i % 10 == 0:
            save_checkpoint(model, optimizer, i, save_path)
    
    