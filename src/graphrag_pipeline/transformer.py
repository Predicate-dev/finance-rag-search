from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - keeps non-training modules usable.
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from graphrag_pipeline.tokenizer import FinancialTokenizer


@dataclass(frozen=True)
class MiniGPTConfig:
    vocab_size: int
    max_seq_len: int = 512
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1


if torch is not None:

    class MultiHeadSelfAttention(nn.Module):
        """Causal multi-head self-attention implemented directly in PyTorch."""

        def __init__(self, config: MiniGPTConfig) -> None:
            super().__init__()
            if config.d_model % config.n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            self.n_heads = config.n_heads
            self.head_dim = config.d_model // config.n_heads
            self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
            self.proj = nn.Linear(config.d_model, config.d_model)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, d_model = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

            scores = q @ k.transpose(-2, -1)
            scores = scores / (self.head_dim**0.5)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            attended = weights @ v
            attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
            return self.proj(attended)


    class FeedForward(nn.Module):
        def __init__(self, config: MiniGPTConfig) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(config.dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class DecoderBlock(nn.Module):
        def __init__(self, config: MiniGPTConfig) -> None:
            super().__init__()
            self.attention_norm = nn.LayerNorm(config.d_model)
            self.attention = MultiHeadSelfAttention(config)
            self.ffn_norm = nn.LayerNorm(config.d_model)
            self.feed_forward = FeedForward(config)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attention(self.attention_norm(x))
            x = x + self.feed_forward(self.ffn_norm(x))
            return x


    class MiniGPT(nn.Module):
        """Decoder-only Transformer suitable for small financial RAG experiments."""

        def __init__(self, config: MiniGPTConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
            self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
            self.norm = nn.LayerNorm(config.d_model)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
            self.dropout = nn.Dropout(config.dropout)
            self.apply(self._init_weights)

        def forward(
            self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            batch, seq_len = input_ids.shape
            if seq_len > self.config.max_seq_len:
                raise ValueError(f"Sequence length {seq_len} exceeds {self.config.max_seq_len}")
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            x = self.dropout(x)
            for block in self.blocks:
                x = block(x)
            logits = self.lm_head(self.norm(x))
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                    labels[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
            return logits, loss

        @torch.no_grad()
        def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int,
            temperature: float = 0.8,
            top_k: int = 50,
            eos_id: int | None = None,
        ) -> torch.Tensor:
            self.eval()
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.max_seq_len :]
                logits, _ = self(context)
                next_logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k > 0:
                    values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < values[:, [-1]]] = float("-inf")
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_id], dim=1)
                if eos_id is not None and bool((next_id == eos_id).all()):
                    break
            return input_ids

        @staticmethod
        def _init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)


    class LanguageModelDataset(Dataset):
        def __init__(self, texts: Iterable[str], tokenizer: FinancialTokenizer, max_seq_len: int) -> None:
            self.examples = [
                tokenizer.encode(text, max_length=max_seq_len)
                for text in texts
                if len(tokenizer.encode(text, max_length=max_seq_len)) > 2
            ]
            self.max_seq_len = max_seq_len
            self.pad_id = tokenizer.pad_id

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> torch.Tensor:
            ids = self.examples[index]
            padded = ids + [self.pad_id] * (self.max_seq_len - len(ids))
            labels = [token_id if token_id != self.pad_id else -100 for token_id in padded]
            return torch.tensor(padded), torch.tensor(labels)


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for the custom Transformer") from TORCH_IMPORT_ERROR


def build_model(config: MiniGPTConfig):
    require_torch()
    return MiniGPT(config)


def load_minigpt_checkpoint(checkpoint_path: str | Path, map_location: str | None = None):
    """Load a trained MiniGPT checkpoint and its tokenizer."""

    require_torch()
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu")
    config = MiniGPTConfig(**checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_path = Path(checkpoint["tokenizer_path"])
    if not tokenizer_path.exists():
        tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    tokenizer = FinancialTokenizer.load(tokenizer_path)
    return model, tokenizer


def train_language_model(
    model,
    tokenizer: FinancialTokenizer,
    texts: list[str],
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    device: str | None = None,
) -> list[float]:
    """Basic next-token training loop for the custom decoder-only Transformer."""

    require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = LanguageModelDataset(texts, tokenizer, model.config.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    losses: list[float] = []

    for _ in range(epochs):
        model.train()
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, labels=labels)
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return losses
