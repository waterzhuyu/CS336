import torch

from torch import Tensor
from jaxtyping import Float, Int64
from cs336_basics.models import softmax

@torch.inference_mode()
def generate(
    model: torch.nn.Module, 
    prompt: Int64[Tensor, "batch seq_len"], 
    max_gen_len: int,
    temperature: float,
    top_p: float,
    eos_id: int
):
    """Do not support batch inference. `batch = 1`"""
    model.eval()

    for _ in range(max_gen_len):

        with torch.no_grad():
            logits = model(prompt) # Float[Tensor, "batch seq_len vocab_size"]

        logits = logits[:, -1, :] # Get last hidden state
        if temperature > 0: 
            # temperature scaling
            logits /= temperature
        
        probs = softmax(logits, dim=-1)

        if top_p < 1:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

            probs[indices_to_remove] = 0.0

            probs = probs / probs.sum(dim=-1, keepdim=True)

        if temperature == 0:
            # greedy sampling 
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            idx_next = torch.multinomial(probs, num_samples=1)

        prompt = torch.cat((prompt, idx_next), dim=1)

        if (idx_next == eos_id).all():
            break
    
    return prompt
