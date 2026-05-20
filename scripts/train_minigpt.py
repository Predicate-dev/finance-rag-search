from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from graphrag_pipeline.tokenizer import FinancialTokenizer
from graphrag_pipeline.transformer import MiniGPTConfig, build_model, train_language_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom MiniGPT finance model.")
    parser.add_argument("--corpus", type=Path, default=Path("data/finance_seed_corpus.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/minigpt_finance"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=3000)
    return parser.parse_args()


def load_training_examples(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    examples = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not examples:
        raise ValueError(f"No training examples found in {path}")
    return examples


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_training_examples(args.corpus)
    tokenizer = FinancialTokenizer()
    tokenizer.fit(examples, vocab_size=args.vocab_size, min_freq=1)

    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = build_model(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    losses = train_language_model(
        model=model,
        tokenizer=tokenizer,
        texts=examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )

    tokenizer_path = args.out_dir / "tokenizer.json"
    checkpoint_path = args.out_dir / "model.pt"
    metrics_path = args.out_dir / "training_metrics.json"

    tokenizer.save(tokenizer_path)
    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": model.cpu().state_dict(),
            "tokenizer_path": str(tokenizer_path),
        },
        checkpoint_path,
    )
    metrics = {
        "examples": len(examples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": device,
        "losses": losses,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved tokenizer: {tokenizer_path}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Initial loss: {metrics['initial_loss']}")
    print(f"Final loss: {metrics['final_loss']}")


if __name__ == "__main__":
    main()
