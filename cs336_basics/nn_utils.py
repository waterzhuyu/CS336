import math
import torch

from torch import Tensor
from torch import nn
from jaxtyping import Float, Int

from typing import Iterable

def cross_entropy(logits: Float[Tensor, "batch seq vocab_size"], targets: Int[Tensor, "batch seq"]) -> Float[Tensor, " ... "]:
    """
    Loss return a mean across batches
    """
    assert logits.shape[:-1] == targets.shape, "logits and targtes should have same shape except the last dimension"

    # Substract max val from logits to keep numerical stability.
    logits -= torch.max(logits, dim=-1, keepdim=True).values

    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))

    logits_for_targets = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    negative_log_softmax = log_sum_exp - logits_for_targets

    return torch.mean(negative_log_softmax)

def gradient_clipping(param_iter: Iterable[nn.Parameter], max_norm: float) -> float:
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

    return total_norm