from dataclasses import dataclass


@dataclass
class GPT:
    name: str
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int


def nearest_multiple_of_64(x: float) -> int:
    return round(x / 64) * 64


def flops_breakdown(V, L, layers, d_model, d_ff):
    """Matmul FLOPs per component for one forward pass over L tokens.

    Everything else is elementwise and not a matmul, so it is left out:
    the embedding is a lookup (V * L * d_model * 2 would be wrong), and
    RMSNorm, RoPE, softmax, SiLU and the residual adds are all O(L * d_model).
    """
    return {
        "qkv_proj": layers * 3 * 2 * L * d_model**2,
        "attn": layers * (2 * L**2 * d_model + 2 * L**2 * d_model),  # QK^T and A @ V
        "out_proj": layers * 2 * L * d_model**2,
        # w1 and w3 are (d_model -> d_ff), w2 is (d_ff -> d_model)
        "swiglu": layers * (2 * (2 * L * d_model * d_ff) + 2 * L * d_model * d_ff),
        "lm_head": 2 * L * d_model * V,
    }


def forward_flops(V, L, layers, d_model, d_ff):
    return sum(flops_breakdown(V, L, layers, d_model, d_ff).values())


def report(model: GPT, context_length: int | None = None) -> int:
    L = context_length if context_length is not None else model.context_length
    parts = flops_breakdown(model.vocab_size, L, model.num_layers, model.d_model, model.d_ff)
    total = sum(parts.values())

    print(f"{model.name} (context_length={L:,}): {total:.4e} FLOPs")
    for name, flops in parts.items():
        print(f"  {name:9s} {flops / 1e9:10.1f} GFLOP  {100 * flops / total:5.1f}%")
    attn_block = parts["qkv_proj"] + parts["attn"] + parts["out_proj"]
    print(f"  {'(attn)':9s} {attn_block / 1e9:10.1f} GFLOP  {100 * attn_block / total:5.1f}%")
    print()
    return total


def make(name, layers, d_model, heads) -> GPT:
    return GPT(
        name=name,
        vocab_size=50257,
        context_length=1024,
        num_layers=layers,
        d_model=d_model,
        num_heads=heads,
        d_ff=nearest_multiple_of_64(d_model * 8 / 3),
    )


if __name__ == "__main__":
    gpt2_small = make("gpt2_small", 12, 768, 12)
    gpt2_medium = make("gpt2_medium", 24, 1024, 16)
    gpt2_large = make("gpt2_large", 36, 1280, 20)
    gpt2_xl = make("gpt2_xl", 48, 1600, 25)

    print("=== question (b) + (c) ===")
    report(gpt2_xl)

    print("=== question (d) ===")
    for model in (gpt2_small, gpt2_medium, gpt2_large, gpt2_xl):
        report(model)

    print("=== question (e) ===")
    short = report(gpt2_xl, context_length=1024)
    long = report(gpt2_xl, context_length=16384)
    print(f"16,384 vs 1,024 tokens: {long / short:.1f}x total FLOPs")
