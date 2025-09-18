import math
import torch

from torch import Tensor
from torch import nn
from jaxtyping import Float, Int

from typing import Iterable

def cross_entropy(logits: Float[Tensor, " batch vocab_size"], targets: Int[Tensor, " batch "]) -> Float[Tensor, " ... "]:
    """
    Loss return a mean across batches
    """
    assert logits.shape[:-1] == targets.shape, "logits and targtes should have same shape except the last dimension"
    assert logits.ndim == 2 and targets.ndim == 1, "logits shape like (batch vocab_size) and targets shape like (batch)"

    # Substract max val from logits to keep numerical stability.
    logits -= torch.max(logits, dim=-1, keepdim=True).values

    negative_log_softmax = torch.log(torch.sum(torch.exp(logits), dim=-1)) - logits[torch.arange(logits.size(0)), targets]

    return torch.mean(negative_log_softmax)

def gradient_clipping(param_iter: Iterable[nn.Parameter], max_norm: float) -> None:
    """
    Clip the gradient norm of an iterable of parameters.

    The norm is computed over the norms of the individual gradients of all parameters, 
    as if the norms of the individual gradients were concatenated into a single vector. 
    Gradients are modified in-place.
    """
    
    total_norm = math.sqrt(
        sum([
            torch.sqrt(torch.sum(param.grad ** 2)) ** 2
            for param in param_iter if param.grad is not None
            ]
        )
    )

    if total_norm > max_norm:
        for param in param_iter:
            if param.grad is not None:
                param.grad.data *= max_norm / (total_norm + 1e-6)
