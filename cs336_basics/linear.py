import numpy as np
import torch
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
        x_column = x.unsqueeze(-1)
        mult = self.W @ x_column
        return mult.squeeze(-1)
        # x_column = rearrange(x, "batch column -> batch column 1")
        # return einsum(x_column, self.W, "batch in, out in -> batch out")
