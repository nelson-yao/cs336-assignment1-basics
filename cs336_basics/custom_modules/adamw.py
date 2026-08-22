from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas, eps=1e-8, weight_decay=1e-2, lr=1e-3):
        # params:
        # {
        #     "m": first moment
        #     "v": second moment
        # }
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):  # noqa: UP045
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            epsilon = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.

                t = state.get("t", 1)
                grad = p.grad.data
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data = p.data * (1 - lr * weight_decay)

                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                m = beta1 * m + grad * (1 - beta1)
                v = beta2 * v + grad**2 * (1 - beta2)
                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)

                state["m"], state["v"], state["t"] = m, v, t + 1

        return loss
