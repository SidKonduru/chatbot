"""Simple retrieval chatbot for coherent replies from chat pairs."""

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class _Pair:
    question: str
    answer: str
    norm_question: str
    question_tokens: set[str]


class RetrievalChatbot:
    """Respond using nearest Q/A pair matching."""

    def __init__(self, pairs: list[_Pair]):
        if not pairs:
            raise ValueError("No Q/A pairs available for retrieval chat.")
        self.pairs = pairs

    @classmethod
    def from_file(cls, chat_path: str):
        if not os.path.exists(chat_path):
            raise FileNotFoundError(f"Missing chat data file: {chat_path}")

        with open(chat_path, encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        if len(raw_lines) < 2:
            raise ValueError("Chat data must contain at least one Q/A pair.")

        if len(raw_lines) % 2 != 0:
            # Ignore the final dangling line if present.
            raw_lines = raw_lines[:-1]

        pairs: list[_Pair] = []
        for i in range(0, len(raw_lines), 2):
            q = raw_lines[i]
            a = raw_lines[i + 1]
            nq = _normalize(q)
            pairs.append(
                _Pair(
                    question=q,
                    answer=a,
                    norm_question=nq,
                    question_tokens=set(nq.split()),
                )
            )
        return cls(pairs)

    def reply(self, question: str) -> str:
        normalized = _normalize(question)
        if not normalized:
            return "Could you rephrase that?"

        tokens = set(normalized.split())

        # Exact lookup first for stable deterministic answers.
        for pair in self.pairs:
            if normalized == pair.norm_question:
                return pair.answer

        best_pair = None
        best_score = -1.0

        for pair in self.pairs:
            if tokens and pair.question_tokens:
                overlap = len(tokens & pair.question_tokens)
                union = len(tokens | pair.question_tokens)
                jaccard = overlap / union if union else 0.0
            else:
                jaccard = 0.0

            fuzz = SequenceMatcher(None, normalized, pair.norm_question).ratio()
            score = max(jaccard, fuzz)
            if score > best_score:
                best_score = score
                best_pair = pair

        if best_pair and best_score >= 0.38:
            return best_pair.answer

        return (
            "I am not fully sure yet. Try asking about greetings, Python, "
            "machine learning, or chatbot basics."
        )
