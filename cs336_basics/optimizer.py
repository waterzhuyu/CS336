import math

from typing import Dict, Optional, Callable, Iterable, Any, Tuple

import torch
from torch.optim import Optimizer
from torch import Tensor

class SGD(Optimizer):
    """
    Args:
        params: collection of parameters to be optimized. (or parameter groups, in case 
            the use want to use different hyperparameters, such as learning rates, for different parts of the model).

            Make sure to pass `params` to the `__init__` method of the base class, which will 
            store these parameters for use in step. You can take additional arguments depending 
            on the optimizer (e.g., the learning rate is a common one), and pass them to the base 
            class constructor as a dictionary, where keys are the names (strings) you choose for
            these parameters.

    """
    def __init__(self, params, lr=1e-3):
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}

        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None) -> Optional[float]: # type: ignore
        """
        Make one update of the parameters. 
        This method should iterate through each parameter tensor `p` and modify them *in place*, 
        i.e. setting `p.data`, which holds the tensor associated with that parameter based on
        the gradient `p.grad` (if it exists), the tensor representing the gradient of the loss 
        with respect to that parameter.  
        """
        loss = None if closure is None else closure()
        # list of params dict
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p
                t = state.get("t", 0) # Get iteration number from the state, or initial value
                grad = p.grad.data # Get the gradient of loss with respect to p
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place
                state["t"] = t + 1 # Increment iteration number
        
        return loss

class AdamW(Optimizer):
    def __init__(
            self, params: Iterable[Tensor] | Iterable[Dict[str, Any]] | Iterable[Tuple[str, Tensor]], 
            # defaults: Dict[str, Any]
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.95),
            eps: float = 1e-8,
            weight_decay: float = 1e-2
        ) -> None:

        if not 0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]: # type: ignore
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Get the state associated with p
                state = self.state[p]

                # Get some maintained variable from the state, or initial value
                t = state.get('t', 1)
                first_momentum = state.get('first_momentum', torch.zeros_like(p))
                second_momentum = state.get('second_momentum', torch.zeros_like(p))

                # Get the gradient
                grad = p.grad.data

                # Update the maintained variable in state
                first_momentum = betas[0] * first_momentum + (1 - betas[0]) * grad
                second_momentum = betas[1] * second_momentum + (1 - betas[1]) * (grad ** 2)

                adjusted_lr = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

                # Update the parameter in-place
                p.data -= adjusted_lr * first_momentum / (torch.sqrt(second_momentum) + eps)
                p.data -= lr * weight_decay * p.data

                # Update the maintained variable in state
                state['first_momentum'] = first_momentum
                state['second_momentum'] = second_momentum
                state['t'] = t + 1

        return loss

def get_lr_cosine_schedule(
        iter: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
    ) -> float:

    # warmup 
    if iter < warmup_iters:
        return iter / warmup_iters * max_learning_rate
    
    # cosine annealing
    if warmup_iters <= iter <= cosine_cycle_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate) / 2 * \
        (1 + math.cos(math.pi * (iter - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    
    # post annealing when `iter > cosine_cycle_iters`
    return min_learning_rate