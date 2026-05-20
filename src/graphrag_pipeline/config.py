from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings shared across the GraphRAG modules."""

    database_path: Path = Path("graphrag.db")
    request_timeout_seconds: float = 10.0
    requests_per_second: float = 0.5
    chunk_token_budget: int = 220
    chunk_overlap_tokens: int = 40
    embedding_dim: int = 384
    vector_top_k: int = 6
    graph_hops: int = 1
    graph_neighbors_per_entity: int = 8
    max_prompt_tokens: int = 512
    generation_tokens: int = 96
    generation_temperature: float = 0.25
    generation_top_k: int = 8
