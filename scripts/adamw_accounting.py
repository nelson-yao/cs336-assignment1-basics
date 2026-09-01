batch_size = 1
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff: int = d_model * 8 // 3


def param_count(
    vocab_size, context_length, num_layers, d_model, num_heads, d_ff
):
    # each transformer block
    rms = 2 * d_model
    attn = 4 * d_model**2
    ff = 3 * d_ff * d_model
    block = rms + attn + ff

    emb = vocab_size * d_model
    output = vocab_size * d_model

    optim = 4 * (block * num_layers + d_model + emb + output)

    return optim


def activation_per_sample(vocab_size, L, num_layers, d_model, h, d_ff):
    activation = (
        (
            2 * d_model * L  # rms
            + L**2 * h  # attn scores
            + h * L**2  # per-head softmax
            + 3 * L * d_model  # QKV projection
            + L * d_model  # weighted sum
            + L * d_model  # output projection
            + d_model * L  # ff
            + 4 * L * d_ff  # SwiLu
        )
        * num_layers
        + L * d_model  # final rms
        + L * vocab_size * 2  # logits and loss
    )
    return activation


def adamw_param_count_per_batch(
    batch_size, vocab_size, context_length, num_layers, d_model, num_heads, d_ff
):
    return (
        param_count(
            vocab_size,
            context_length,
            num_layers,
            d_model,
            num_heads,
            d_ff,
        )
        + activation_per_sample(
            vocab_size,
            context_length,
            num_layers,
            d_model,
            num_heads,
            d_ff,
        )
        * batch_size
    )


def max_batch_size(
    gpu_mem, vocab_size, context_length, num_layers, d_model, num_heads, d_ff
):
    gpu_mem -= (
        param_count(
            vocab_size,
            context_length,
            num_layers,
            d_model,
            num_heads,
            d_ff,
        )
        * 4
    )

    max_batch_size = int(
        gpu_mem
        / (
            activation_per_sample(
                vocab_size,
                context_length,
                num_layers,
                d_model,
                num_heads,
                d_ff,
            )
            * 4
        )
    )

    return max_batch_size


def adamw_flop_per_step(n_parameter):
    """
    calculation per step:
    lr * weight_decay
    1 - beta1
    1 - beta2
    adjust alpha_t : 7

    per param:

    apply weight decay: 2
    update first moment: 3
    update second moment: 4
    Apply moment-adjusted weight updates : 5
    """
    return 14 * n_parameter + 10


if __name__ == "__main__":
    peak_param = adamw_param_count_per_batch(
        batch_size,
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
    )

    print(
        f"Peak memory {peak_param * 4 / 10e9} GB",
    )

    gpu_mem = 80 * 10e9
    max_size = max_batch_size(
        gpu_mem,
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
    )
    print("Maximum batch size :", max_size)
