# from pretokenization_multiprocess import pretokenize_and_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = 8


# def train_bpe_chunk(chunk: str) -> dict[str, int]:
#     pretoken_counts = pretokenize_and_count(chunk)


def train_bpe(
    input_path: str, vocab_size: str, special_tokens=None
) -> tuple[dict[str, bytes], list[tuple[bytes, bytes]]]:
    pass
    # with open(..., "rb") as f:
    #     num_processes = 4
    #     boundaries = find_chunk_boundaries(f, num_processes, special_tokens)

    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #         # Run pre-tokenization on your chunk and store the counts for each pre-token
    #         process = Process(target=pretokenize_and_count, args=(chunk,))
    #         process.start()


def merge_bpe():
    pass
