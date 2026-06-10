import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum
from torch import nn


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
        print("selection shape :", selection.shape)
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
