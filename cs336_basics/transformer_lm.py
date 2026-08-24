from cs336_basics.modules import RMSNorm
import torch
from cs336_basics.modules import TransformerBlock
from cs336_basics.modules import Embedding
from cs336_basics.modules import Linear


class TransformerLM(torch.Module):
    def __init__(
        self,
        embedding: Embedding,
        blocks: list[TransformerBlock],
        rms_final: RMSNorm,
        lm_head: Linear,
    ):
        super().__init__()
        self.embed_layer = embedding
        self.blocks = blocks
        self.rms_final = rms_final
        self.lm_head = lm_head

    def forward(self, x: torch.Tensor):
        x = self.embed_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.rms_final(x)
        return self.lm_head(x)
