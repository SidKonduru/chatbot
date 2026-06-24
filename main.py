#!/usr/bin/env python3
"""CLI entry point for the seq2seq chatbot."""

import argparse
import os
import sys

import data_preprocessing
from retrieval_chat import RetrievalChatbot
from seq2seq_model import Seq2SeqModel


def _data_dir(path):
    return path if path.endswith(os.sep) else path + os.sep


def cmd_preprocess(args):
    data_preprocessing.process_data(data_dir=args.data_dir)


def cmd_train(args):
    data_path = _data_dir(args.data_dir)
    metadata, idx_q, idx_a = data_preprocessing.load_data(data_path)

    (train_x, train_y), _, (valid_x, valid_y) = data_utils.split_dataset(
        idx_q, idx_a, seed=args.seed
    )

    model = Seq2SeqModel(
        enc_max_len=train_x.shape[1],
        dec_max_len=train_y.shape[1],
        vocab_size=len(metadata["idx2w"]),
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        ckpt_dir=args.weights_dir,
    )

    print(
        f"Training on {len(train_x)} samples, validating on {len(valid_x)} samples "
        f"(vocab={len(metadata['idx2w'])}, emb_dim={args.emb_dim}, layers={args.num_layers})"
    )

    model.train(
        train_x,
        train_y,
        valid_x,
        valid_y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    print(f"Training complete. Model saved to {args.weights_dir}/")


def cmd_chat(args):
    retriever = RetrievalChatbot.from_file(args.chat_data)

    print("Chatbot ready. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        answer = retriever.reply(question)
        print(f"Bot: {answer}")


def build_parser():
    parser = argparse.ArgumentParser(description="Seq2seq chatbot")
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing metadata.pkl and idx_*.npy (default: project root)",
    )
    parser.add_argument(
        "--weights-dir",
        default="weights",
        help="Directory for model checkpoints (default: weights)",
    )
    parser.add_argument(
        "--chat-data",
        default="data/chat.txt",
        help="Q/A text file for retrieval chat (default: data/chat.txt)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preprocess", help="Build dataset from data/chat.txt")

    train_parser = subparsers.add_parser("train", help="Train the seq2seq model")
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--patience", type=int, default=5)
    train_parser.add_argument("--emb-dim", type=int, default=512)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--seed", type=int, default=42)

    subparsers.add_parser("chat", help="Chat with the trained model")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "chat":
        cmd_chat(args)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
