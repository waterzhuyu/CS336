import os
from typing import IO, BinaryIO

import torch
from torch import nn
from torch.optim import Optimizer

def save_checkpoint(
        model: nn.Module, 
        optimizer: Optimizer, 
        iteration: int, 
        out_path: str | os.PathLike | BinaryIO | IO[bytes]
    ) -> None:

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    state = {
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'iteration': iteration
    }

    torch.save(state, out_path)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes], 
        model: nn.Module, 
        optimizer: Optimizer
    ) -> int:
    state = torch.load(src)

    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])

    return state['iteration']