from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Article:
    source_url: str
    title: str
    body: str
    published_at: datetime | None
    source: str
    id: str | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class TextChunk:
    article_id: str
    chunk_index: int
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True)
class Entity:
    name: str
    entity_type: str
    canonical_id: str
    confidence: float = 1.0
    sentiment: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    source: Entity
    target: Entity
    relation_type: str
    confidence: float
    sentiment: float
    evidence: str
    article_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphNeighbor:
    source_name: str
    relation_type: str
    target_name: str
    sentiment: float
    evidence: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    article_id: str
    title: str
    source_url: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedContext:
    query: str
    chunks: list[RetrievedChunk]
    graph_neighbors: list[GraphNeighbor]
    query_entities: list[Entity]


@dataclass(frozen=True)
class SentimentBreakdown:
    overall: float
    by_entity: dict[str, float]
    by_relation: dict[str, float]
