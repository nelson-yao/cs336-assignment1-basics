import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

# lt.monkey_patch()  # Overrides default torch.Tensor print behavior


def initialize(*size, dtype=None, device=None, a=-3.0, b=3.0):
    return nn.Parameter(
        torch.nn.init.trunc_normal_(
            torch.empty(
                *size,
                device=device,
                dtype=dtype,
            ),
            std=np.sqrt(2 / sum([x for x in size])),
            a=-3.0,
            b=3.0,
        )
    )


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    out_features,
                    in_features,
                    device=device,
                    dtype=dtype,
                ),
                std=np.sqrt(2 / (in_features + out_features)),
                a=-3.0,
                b=3.0,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_column = x.unsqueeze(-1)
        # mult = self.W @ x_column
        # return mult.squeeze(-1)
        # x_column = rearrange(x, "batch column -> batch column 1")
        return einsum(x, self.W, "... in, out in -> ... out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    num_embeddings,
                    embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                std=np.sqrt(2 / (num_embeddings + embedding_dim)),
                a=-3.0,
                b=3.0,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        selection = F.one_hot(token_ids, num_classes=self.num_embeddings).to(self.emb.dtype)
        return einsum(selection, self.emb, "batch sequence vocab, vocab d -> batch sequence  d")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.g = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    d_model,
                    device=device,
                    dtype=dtype,
                ),
                std=np.sqrt(2 / (d_model)),
                a=-3.0,
                b=3.0,
            )
        )
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.sum(torch.square(x), dim=-1) / self.d_model + self.eps)

        result = x / rms.unsqueeze(-1) * self.g
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_ff: int,
        d_model: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.w1 = initialize(d_ff, d_model)
        self.w2 = initialize(d_model, d_ff)
        self.w3 = initialize(d_ff, d_model)

    def forward(self, x: Float[Tensor, " ... d_model"]):
        w1x = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3x = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1x * torch.sigmoid(w1x)
        return einsum(self.w2, (silu * w3x), "d_model d_ff, ... d_ff -> ... d_model")


class RoPE(nn.Module):
    # https://github.com/zhasion/CS336/blob/main/assignment1-basics/cs336_basics/module.py#L108
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        denominator = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        seq_vec = torch.arange(max_seq_len).to(torch.float32)
        angles = torch.repeat_interleave(torch.outer(seq_vec, denominator), 2, dim=-1)

        self.d = d_k
        self.d_vec = torch.arange(0, d_k, 2)
        self.max_seq_len = max_seq_len

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos_cached[token_positions]
        sin_pos = self.sin_cached[token_positions]

        x_2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)

        x_2 = x_2.flatten(start_dim=-2)

        return x * cos_pos + x_2 * sin_pos


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.amax(dim=dim, keepdim=True)
    return x.exp() / x.exp().sum(dim=dim, keepdim=True)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = key.shape[-1]
    pre = einsum(query, key, "... q d_k, ... k d_k -> ... q k") / np.sqrt(d_k)

    pre = pre.masked_fill(~mask, float("-inf"))
    soft = softmax(pre, dim=-1)
    output = einsum(soft, value, "... q k, ... k d_v-> ... q d_v")

    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, h: int, max_seq_len: int, theta=10000, device=None):
        super().__init__()
        d_k = d_v = int(d_model / h)
        self.d_model = d_model
        self.h = h
        self.WQ = initialize(h * d_k, d_model, device=device)
        self.WK = initialize(h * d_k, d_model, device=device)
        self.WV = initialize(h * d_v, d_model, device=device)
        self.WO = initialize(d_model, h * d_v, device=device)
        self.max_seq_len = max_seq_len
        self.rope = RoPE(theta, d_k, max_seq_len, device=device)

    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        q = rearrange(in_features @ self.WQ.T, "... seq (h d_k) -> ... h seq d_k", h=self.h)
        if token_positions is not None:
            q = self.rope(q, token_positions)
        k = rearrange(in_features @ self.WK.T, "... seq (h d_k) -> ... h seq d_k", h=self.h)
        if token_positions is not None:
            k = self.rope(k, token_positions)
        v = rearrange(in_features @ self.WV.T, "... seq  (h d_v) -> ... h seq d_v", h=self.h)
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=0) > 0.5
        attn: Tensor = attention(q, k, v, mask=mask)
        attn = rearrange(attn, "... h seq d_v -> ... seq  (h d_v)")
        return attn @ self.WO.T
