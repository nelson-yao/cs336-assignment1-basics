import math
from collections.abc import Iterable
import torch


def lr_schedule(t, amax, amin, tw, tc):
    if t < tw:
        return t / tw * amax
    elif t < tc:
        return amin + 0.5 * (1 + math.cos((t - tw) / (tc - tw) * math.pi)) * (
            amax - amin
        )
    else:
        return amin


def clip_gradient(params: Iterable[torch.nn.Parameter], M):
    grads = [(p, torch.clone(p.grad)) for p in params if p.grad is not None]
    norm = math.sqrt(sum([torch.sum(torch.square(grad)) for p, grad in grads]))
    if norm >= M:
        for p, grad in grads:
            p.grad = M / (norm + 1e-6) * grad
