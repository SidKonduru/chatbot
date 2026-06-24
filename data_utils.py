"""Shared data utilities for preprocessing, training, and inference."""

import numpy as np

EN_WHITELIST = "0123456789abcdefghijklmnopqrstuvwxyz "


def pad_seq(seq, lookup, maxlen, unk_token="unk"):
    """Map words to indices and pad to maxlen with index 0."""
    indices = []
    unk_idx = lookup.get(unk_token, 1)
    for word in seq:
        indices.append(lookup.get(word, unk_idx))
    return indices + [0] * (maxlen - len(seq))


def decode(sequence, lookup, separator=" "):
    """Convert token indices back to a string, skipping padding (index 0)."""
    return separator.join(lookup[element] for element in sequence if element)


def encode(sentence, lookup, maxlen, whitelist=EN_WHITELIST):
    """Normalize, tokenize, and encode a user question for inference."""
    sentence = sentence.lower()
    sentence = "".join(ch for ch in sentence if ch in whitelist)
    tokens = sentence.strip().split(" ")
    if len(tokens) > maxlen:
        tokens = tokens[-maxlen:]
    return np.array(pad_seq(tokens, lookup, maxlen), dtype=np.int32)


def split_dataset(x, y, ratio=(0.7, 0.15, 0.15), seed=42):
    """Shuffle and split into train, test, and validation sets."""
    data_len = len(x)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(data_len)

    train_end = int(data_len * ratio[0])
    test_end = train_end + int(data_len * ratio[1])

    train_idx = indices[:train_end]
    test_idx = indices[train_end:test_end]
    valid_idx = indices[test_end:]

    return (
        (x[train_idx], y[train_idx]),
        (x[test_idx], y[test_idx]),
        (x[valid_idx], y[valid_idx]),
    )


def batch_gen(x, y, batch_size):
    """Yield consecutive batches of (X, Y) arrays."""
    n = len(x)
    while True:
        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n:
                continue
            yield x[start:end], y[start:end]


def rand_batch_gen(x, y, batch_size, seed=42):
    """Yield random batches of (X, Y) arrays."""
    rng = np.random.default_rng(seed)
    n = len(x)
    while True:
        sample_idx = rng.choice(n, size=batch_size, replace=False)
        yield x[sample_idx], y[sample_idx]


def prepare_decoder_inputs(targets):
    """Shift targets right for teacher forcing (GO token = index 0)."""
    decoder_input = np.zeros_like(targets)
    decoder_input[:, 1:] = targets[:, :-1]
    return decoder_input
