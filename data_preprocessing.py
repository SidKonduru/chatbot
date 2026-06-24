"""Prepare chat data from raw text into indexed numpy arrays."""

import itertools
import os
import pickle

import nltk
import numpy as np

from data_utils import EN_WHITELIST, pad_seq

FILENAME = "data/chat.txt"

limit = {
    "maxq": 20,
    "minq": 0,
    "maxa": 20,
    "mina": 3,
}

UNK = "unk"
VOCAB_SIZE = 6000


def read_lines(filename):
    with open(filename, encoding="utf-8") as f:
        return f.read().split("\n")[:-1]


def filter_line(line, whitelist):
    return "".join(ch for ch in line if ch in whitelist)


def index_(tokenized_sentences, vocab_size):
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    index2word = ["_", UNK] + [word for word, _ in vocab]
    word2index = {word: idx for idx, word in enumerate(index2word)}
    return index2word, word2index, freq_dist


def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences) // 2

    for i in range(0, len(sequences), 2):
        qlen = len(sequences[i].split(" "))
        alen = len(sequences[i + 1].split(" "))
        if limit["minq"] <= qlen <= limit["maxq"] and limit["mina"] <= alen <= limit["maxa"]:
            filtered_q.append(sequences[i])
            filtered_a.append(sequences[i + 1])

    filt_data_len = len(filtered_q)
    filtered_pct = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
    print(f"{filtered_pct}% filtered from original data")
    return filtered_q, filtered_a


def zero_pad(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)
    idx_q = np.zeros([data_len, limit["maxq"]], dtype=np.int32)
    idx_a = np.zeros([data_len, limit["maxa"]], dtype=np.int32)

    for i in range(data_len):
        idx_q[i] = pad_seq(qtokenized[i], w2idx, limit["maxq"], unk_token=UNK)
        idx_a[i] = pad_seq(atokenized[i], w2idx, limit["maxa"], unk_token=UNK)

    return idx_q, idx_a


def process_data(data_dir="."):
    filepath = os.path.join(data_dir, FILENAME)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Missing {filepath}. Add a question/answer dataset (one line per utterance, "
            "alternating question then answer)."
        )

    print("\n>> Read lines from file")
    lines = read_lines(filepath)
    lines = [line.lower() for line in lines]

    print("\n>> Filter lines")
    lines = [filter_line(line, EN_WHITELIST) for line in lines]

    print("\n>> Filter by sequence length")
    qlines, alines = filter_data(lines)

    print("\n>> Segment lines into words")
    qtokenized = [wordlist.split(" ") for wordlist in qlines]
    atokenized = [wordlist.split(" ") for wordlist in alines]

    print("\n>> Index words")
    idx2w, w2idx, freq_dist = index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    print("\n>> Zero padding")
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print("\n>> Save arrays and metadata")
    np.save(os.path.join(data_dir, "idx_q.npy"), idx_q)
    np.save(os.path.join(data_dir, "idx_a.npy"), idx_a)

    metadata = {
        "w2idx": w2idx,
        "idx2w": idx2w,
        "limit": limit,
        "freq_dist": freq_dist,
    }
    with open(os.path.join(data_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved {len(idx_q)} samples to {data_dir}")


def load_data(path=""):
    with open(path + "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    idx_q = np.load(path + "idx_q.npy")
    idx_a = np.load(path + "idx_a.npy")
    return metadata, idx_q, idx_a


if __name__ == "__main__":
    process_data()
