import math
from collections.abc import Iterable
import torch, random
import numpy as np


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


def get_batch(input_ids, batch_size, context_length, device):
    samples = []
    targets = []
    for i in range(batch_size):
        start = random.randint(0, len(input_ids) - context_length - 1)
        samples.append(input_ids[start : start + context_length])
        targets.append(input_ids[start + 1 : start + context_length + 1])

    sample_tensor = torch.tensor(np.stack(samples)).to(device)
    target_tensor = torch.tensor(np.stack(targets)).to(device)

    return sample_tensor, target_tensor
