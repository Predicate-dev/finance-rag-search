from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


class FinancialTokenizer:
    """Small tokenizer specialized for finance text and RAG prompts."""

    TOKEN_RE = re.compile(
        r"<[A-Z_]+>|[$]?[A-Z]{1,5}\b|[-+]?\d+(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9&.\-']*|[^\s]"
    )
    SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<CONTEXT>", "<GRAPH>", "<QUESTION>", "<ANSWER>"]

    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        self.token_to_id = token_to_id or {
            token: index for index, token in enumerate(self.SPECIAL_TOKENS)
        }
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def fit(self, texts: list[str], vocab_size: int = 12000, min_freq: int = 1) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        for token, freq in counter.most_common(max(0, vocab_size - len(self.SPECIAL_TOKENS))):
            if freq < min_freq:
                continue
            if token not in self.token_to_id:
                index = len(self.token_to_id)
                self.token_to_id[token] = index
                self.id_to_token[index] = token

    def tokenize(self, text: str) -> list[str]:
        return self.TOKEN_RE.findall(text)

    def encode(self, text: str, add_special_tokens: bool = True, max_length: int | None = None) -> list[int]:
        ids = [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in self.tokenize(text)]
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
            if add_special_tokens and ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in ids:
            token = self.id_to_token.get(int(token_id), "<UNK>")
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)
        text = " ".join(tokens)
        return re.sub(r"\s+([.,!?;:%])", r"\1", text)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FinancialTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({str(token): int(index) for token, index in data.items()})
