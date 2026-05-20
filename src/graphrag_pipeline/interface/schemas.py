from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ArticleIngestRequest:
    source_url: str
    title: str
    body: str
    published_at: datetime | None = None
    source: str = "api"

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ArticleIngestRequest":
        return cls(
            source_url=str(value["source_url"]),
            title=str(value["title"]),
            body=str(value["body"]),
            published_at=parse_datetime(value.get("published_at")),
            source=str(value.get("source", "api")),
        )


@dataclass(frozen=True)
class ArticleResult:
    id: str | None
    source_url: str
    title: str
    body_preview: str
    published_at: datetime | None
    source: str


@dataclass(frozen=True)
class IngestArticleResponse:
    article_id: str
    indexed: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_jsonable(self)


@dataclass(frozen=True)
class SeedDemoResponse:
    indexed_article_ids: list[str]
    total_articles: int

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_jsonable(self)


@dataclass(frozen=True)
class SearchFilters:
    tickers: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    published_after: datetime | None = None
    published_before: datetime | None = None
    min_sentiment: float | None = None
    max_sentiment: float | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "SearchFilters":
        value = value or {}
        return cls(
            tickers=list(value.get("tickers") or []),
            entities=list(value.get("entities") or []),
            sources=list(value.get("sources") or []),
            published_after=parse_datetime(value.get("published_after") or value.get("min_published_at")),
            published_before=parse_datetime(
                value.get("published_before") or value.get("max_published_at")
            ),
            min_sentiment=value.get("min_sentiment"),
            max_sentiment=value.get("max_sentiment"),
        )


@dataclass(frozen=True)
class RetrievalOptions:
    vector_top_k: int | None = None
    graph_neighbors_per_entity: int | None = None
    graph_hops: int = 1
    include_graph: bool = True
    include_chunks: bool = True
    include_sentiment: bool = True

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "RetrievalOptions":
        value = value or {}
        return cls(
            vector_top_k=value.get("vector_top_k") or value.get("top_k"),
            graph_neighbors_per_entity=value.get("graph_neighbors_per_entity")
            or value.get("graph_neighbors"),
            graph_hops=int(value.get("graph_hops", 1)),
            include_graph=bool(value.get("include_graph", True)),
            include_chunks=bool(value.get("include_chunks", True)),
            include_sentiment=bool(value.get("include_sentiment", True)),
        )


@dataclass(frozen=True)
class GenerationOptions:
    max_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "GenerationOptions":
        value = value or {}
        return cls(
            max_tokens=value.get("max_tokens"),
            temperature=value.get("temperature"),
            top_k=value.get("top_k"),
        )


@dataclass(frozen=True)
class PromptQueryRequest:
    query: str
    template: str = "financial_rag"
    filters: SearchFilters = field(default_factory=SearchFilters)
    retrieval: RetrievalOptions = field(default_factory=RetrievalOptions)
    generation: GenerationOptions = field(default_factory=GenerationOptions)
    debug: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "PromptQueryRequest":
        return cls(
            query=str(value["query"]),
            template=str(value.get("template", "financial_rag")),
            filters=SearchFilters.from_mapping(value.get("filters")),
            retrieval=RetrievalOptions.from_mapping(value.get("retrieval")),
            generation=GenerationOptions.from_mapping(value.get("generation")),
            debug=bool(value.get("debug", False)),
        )


@dataclass(frozen=True)
class ChunkResult:
    article_id: str
    title: str
    source_url: str
    text: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GraphNeighborResult:
    source_name: str
    relation_type: str
    target_name: str
    sentiment: float
    evidence: str | None


@dataclass(frozen=True)
class EntityResult:
    name: str
    entity_type: str
    canonical_id: str
    sentiment: float
    confidence: float


@dataclass(frozen=True)
class SentimentResult:
    overall: float | None
    entities: dict[str, float]
    relations: dict[str, float]


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[ChunkResult]
    graph_neighbors: list[GraphNeighborResult]
    query_entities: list[EntityResult]


@dataclass(frozen=True)
class PromptResult:
    template: str
    rendered: str | None


@dataclass(frozen=True)
class SearchGenerationResponse:
    answer: str
    sentiment: SentimentResult | None
    retrieval: RetrievalResult
    prompt: PromptResult

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_jsonable(self)


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def dataclass_to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: dataclass_to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [dataclass_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_jsonable(item) for key, item in value.items()}
    return value
