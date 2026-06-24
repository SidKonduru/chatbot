#!/usr/bin/env python3
"""MCP server exposing chatbot tools to Cursor."""

import os
import sys

# MCP uses stdout for the protocol; keep imports and logs off stdout.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

import data_preprocessing
import data_utils
from retrieval_chat import RetrievalChatbot
from seq2seq_model import Seq2SeqModel

mcp = FastMCP(
    "chatbot",
    instructions=(
        "Tools for the local seq2seq chatbot project. "
        "Use chat to get bot replies, train to fit the model, "
        "and status to check whether data and weights are ready."
    ),
)

_MODEL = None
_METADATA = None
_RETRIEVER = None


def _project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _load_metadata():
    global _METADATA
    if _METADATA is None:
        _METADATA, _, _ = data_preprocessing.load_data(_project_root() + os.sep)
    return _METADATA


def _load_model():
    global _MODEL
    if _MODEL is None:
        weights_dir = os.path.join(_project_root(), "weights")
        _MODEL = Seq2SeqModel.load(weights_dir)
    return _MODEL


def _load_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        chat_path = os.path.join(_project_root(), "data", "chat.txt")
        _RETRIEVER = RetrievalChatbot.from_file(chat_path)
    return _RETRIEVER


@mcp.tool()
def chat(question: str) -> str:
    """Send a message and return a coherent retrieval-based reply."""
    retriever = _load_retriever()
    return retriever.reply(question)


@mcp.tool()
def status() -> str:
    """Report whether dataset files and trained weights are available."""
    root = _project_root()
    checks = {
        "metadata.pkl": os.path.exists(os.path.join(root, "metadata.pkl")),
        "idx_q.npy": os.path.exists(os.path.join(root, "idx_q.npy")),
        "idx_a.npy": os.path.exists(os.path.join(root, "idx_a.npy")),
        "weights/best_model.keras": os.path.exists(
            os.path.join(root, "weights", "best_model.keras")
        ),
        "data/chat.txt": os.path.exists(os.path.join(root, "data", "chat.txt")),
    }
    lines = [f"{name}: {'ready' if ok else 'missing'}" for name, ok in checks.items()]
    return "\n".join(lines)


@mcp.tool()
def train(
    epochs: int = 50,
    batch_size: int = 64,
    emb_dim: int = 512,
    num_layers: int = 2,
    patience: int = 5,
) -> str:
    """Train the seq2seq model on the preprocessed dataset."""
    root = _project_root()
    data_path = root + os.sep
    metadata, idx_q, idx_a = data_preprocessing.load_data(data_path)
    (train_x, train_y), _, (valid_x, valid_y) = data_utils.split_dataset(
        idx_q, idx_a
    )

    weights_dir = os.path.join(root, "weights")
    model = Seq2SeqModel(
        enc_max_len=train_x.shape[1],
        dec_max_len=train_y.shape[1],
        vocab_size=len(metadata["idx2w"]),
        emb_dim=emb_dim,
        num_layers=num_layers,
        ckpt_dir=weights_dir,
    )
    model.train(
        train_x,
        train_y,
        valid_x,
        valid_y,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
    )

    global _MODEL
    _MODEL = model
    return f"Training complete. Model saved to {weights_dir}/"


if __name__ == "__main__":
    mcp.run()
