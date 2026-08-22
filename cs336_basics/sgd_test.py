from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):  # noqa: UP045
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or 0.
                grad = (
                    p.grad.data
                )  # Get the gradient of loss with respect to p.
                p.data -= (
                    lr / math.sqrt(t + 1) * grad
                )  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


def run_sgd(lr: float, num_steps: int = 100, seed: int = 0) -> list[float]:
    """Run SGD on a fixed quadratic loss and return the per-step losses.

    A fixed seed re-initializes the weights identically on every call so
    trajectories for different learning rates are directly comparable.
    """
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses = []
    for _ in range(num_steps):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return losses


def plot_loss_trajectories(
    learning_rates: Iterable[float],
    num_steps: int = 100,
    seed: int = 0,
    save_path: str = "sgd_loss_trajectories.png",
) -> str:
    """Run SGD for each learning rate and plot all loss trajectories together."""
    import matplotlib

    matplotlib.use("Agg")  # Headless-safe backend.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))
    for lr in learning_rates:
        losses = run_sgd(lr, num_steps=num_steps, seed=seed)
        plt.plot(range(num_steps), losses, label=f"lr={lr:g}")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")  # Losses span orders of magnitude across LRs.
    plt.title(f"SGD loss trajectories over {num_steps} steps")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")
    return save_path


if __name__ == "__main__":
    learning_rates = [1e-3, 1e-2, 1e-1, 1.0, 1e1]
    plot_loss_trajectories(learning_rates)
