import math

import torch

from torch import nn
from torch import Tensor

from jaxtyping import Float
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ) -> None:
        super().__init__()

        w = torch.empty(in_features, out_features, dtype=dtype, device=device)
        dev = math.sqrt(2 / (in_features + out_features))
        # init to \mathcal{N}(0, 2 / (in_features + out_features)), truncate to (-3\sigma, 3\sigma)
        nn.init.trunc_normal_(w, mean=0, std=dev, a=-3*dev, b=3*dev)

        self.weight = nn.Parameter(w)

    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
        return x @ self.weight

class Embedding(nn.Module):
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            device: torch.device | None = None, 
            dtype=None
        ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        p = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        # init to \mathcal{N}(0, 1), truncate to (-3, 3)
        nn.init.trunc_normal_(p, mean=0, std=1, a=-3, b=-3)

        self.param = nn.Parameter(p)
    
    def forward(self, x: Float[Tensor, "batch seq"]) -> Float[Tensor, "batch seq d_model"]:
        return self.param[x]

class RMSNorm(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            eps: float = 1e-5, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ) -> None:
        super().__init__()

        self.eps = eps

        # init to 1
        g = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(g)
    
    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        in_dtype = x.dtype

        # 'cause square the input, so upcast to float32 to prevent overflow
        x = x.to(torch.float32)

        # Compute Root-Mean-Square along the last dimension.
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x * self.weight / rms

        return result.to(in_dtype)

def swish(x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
    return x / (1 + torch.exp(-x))

class PositionWiseFFN(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            d_ff: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ) -> None:
        """SwiGLU activations"""
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        """W2(SiLU(W1x) odot W2x)"""
        return self.w2(swish(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self, 
            theta: float, 
            d_k: int, 
            max_seq_len: int,
            device: torch.device | None = None
        ) -> None:
        super().__init__()

        position = torch.arange(0, max_seq_len, device=device).unsqueeze(1)

        div_term = torch.exp(
            - torch.arange(0, d_k, 2, device=device) * math.log(theta) / d_k
        )
        div_term = torch.repeat_interleave(div_term, repeats=2) # repeat every item twice

        cos_term = torch.cos(position * div_term) # "max_seq_len d_k"
        sin_term = torch.sin(position * div_term) # "max_seq_len d_k"

        # Cache the computed cos and sin
        self.register_buffer("cos_term", cos_term)
        self.register_buffer("sin_term", sin_term)

    def forward(
            self, 
            x: Float[Tensor, "... seq d_k"], 
            token_positions: Float[Tensor, "... seq"]
        ) -> Float[Tensor, "... seq d_k"]:

        cos_ = self.cos_term[token_positions, :] # type: ignore
        sin_ = self.sin_term[token_positions, :] # type: ignore

        # Construct [-x1, x0, -x3, x2, ... -x(d-1), x(d-2)]
        # d_k = x.shape[-1]
        # idx = torch.stack((torch.arange(1, d_k+1, 2), torch.arange(0, d_k, 2)), dim=1).reshape(-1)
        # rotated_x = x[..., idx]
        # rotated_x[..., 0::2] = -rotated_x[..., 0::2]

        x_odd, x_even = x[..., 1::2], x[..., 0::2]
        rotated_x = torch.stack((-x_odd, x_even), dim=-1).reshape(*x.shape)

        return x * cos_ + rotated_x * sin_

def softmax(input: Tensor, dim: int) -> Tensor:
    """Substract the max val to avoid numercial overflow, keep numerical stability"""
    input = input - torch.max(input, dim=dim, keepdim=True).values

    return torch.exp(input) / torch.sum(torch.exp(input), dim=dim, keepdim=True)

def scaled_dot_product_attention(
        queries: Float[Tensor, "batch ... seq1 d_k"], 
        keys: Float[Tensor, "batch ... seq2 d_k"], 
        values: Float[Tensor, "batch ... seq2 d_v"], 
        mask: Float[Tensor, "batch seq1 seq2"] | None = None
        ) -> Float[Tensor, "batch ... seq1 d_v"]:
    """
    Scaled Dot Product Attention.

    Attention mechanism don't need query and key has the same seqence length.

    mask is a boolean tenosr, `True` at (i, j) means query i does attend to key j, `False` means 
    query i does not attend to key j. This is contrary to convention, which use `True` mean mask this value

    Args:
        keys:
        queries:
        values:
        mask (Float[Tenosr, "batch seq1 seq2"]): a boolean Tensor, `True` at (i, j) means query i does attend to key j
            This is opposed to convertion.
    """
    
    attention_scores = einsum(queries, keys, "... seq1 d_k, ... seq2 d_k -> ... seq1 seq2")
    d_k = keys.shape[-1]
    attention_scores /= math.sqrt(d_k)
    if mask is not None:
        # replace index where is `False` in mask to `-inf`
        # attention_score_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf)
        # attention_scores += attention_score_mask
        attention_scores.masked_fill_(~mask, -torch.inf)
    attention_scores = softmax(attention_scores, dim=-1)

    result = einsum(attention_scores, values, "... seq1 seq2, ... seq2 d_v -> ... seq1 d_v")
    
    return result

class CausalMultiHeadAttention(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            use_rope=False, 
            theta: float | None = None, 
            max_seq_len: int | None = None
        ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.use_rope = use_rope

        assert d_model  % num_heads == 0, "only support num_heads is divisible by d_model"
        d_k = d_model // num_heads
        d_v = d_k

        self.q_proj = Linear(d_model, num_heads * d_k)
        self.k_proj = Linear(d_model, num_heads * d_k)
        self.v_proj = Linear(d_model, num_heads * d_v)
        self.output_proj = Linear(num_heads * d_v, d_model)

        if self.use_rope:
            assert theta is not None
            assert max_seq_len is not None

            self.pe = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    
    def forward(
            self, 
            x: Float[Tensor, "batch seq d_model"], 
            token_positions: Float[Tensor, "... seq"] | None = None
        ) -> Float[Tensor, "batch seq d_model"]:
        b, s, d = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        queries = rearrange(Q, "batch seq (head d_k) -> batch head seq d_k", head=self.num_heads)
        keys = rearrange(K, "batch seq (head d_k) -> batch head seq d_k", head=self.num_heads)
        values = rearrange(V, "batch seq (head d_v) -> batch head seq d_v", head=self.num_heads)

        mask = ~torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
        # mask = mask.unsqueeze(0).unsqueeze(0) # optional, because mask support broadcasting

        if self.use_rope:
            assert token_positions is not None

            queries = self.pe(queries, token_positions)
            keys = self.pe(keys, token_positions)

        result = scaled_dot_product_attention(queries, keys, values, mask) # Float[Tensor, "batch head seq d_v"]
        
        result = rearrange(result, "batch head seq d_v -> batch seq (head d_v)")

        return self.output_proj(result)

class TransformerBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        pass

