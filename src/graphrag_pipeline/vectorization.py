from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from graphrag_pipeline.domain import Article, RetrievedContext, TextChunk
from graphrag_pipeline.entities import FinancialEntityExtractor, split_sentences
from graphrag_pipeline.storage import stable_hash


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.\-']*|[$]?[A-Z]{1,5}\b|[-+]?\d+(?:\.\d+)?%?")


class FinancialTextChunker:
    """Sentence-aware chunking tuned for compact finance news articles."""

    def __init__(self, token_budget: int = 220, overlap_tokens: int = 40) -> None:
        if overlap_tokens >= token_budget:
            raise ValueError("overlap_tokens must be smaller than token_budget")
        self.token_budget = token_budget
        self.overlap_tokens = overlap_tokens

    def chunk_article(self, article: Article, article_id: str) -> list[TextChunk]:
        sentences = split_sentences(f"{article.title}. {article.body}")
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(tokenize(sentence))
            if current and current_tokens + sentence_tokens > self.token_budget:
                chunks.append(" ".join(current))
                overlap = tail_tokens(" ".join(current), self.overlap_tokens)
                current = [overlap] if overlap else []
                current_tokens = len(tokenize(overlap))
            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            chunks.append(" ".join(current))

        return [
            TextChunk(
                article_id=article_id,
                chunk_index=index,
                text=chunk,
                metadata={
                    "source_url": article.source_url,
                    "title": article.title,
                    "source": article.source,
                    "published_at": article.published_at.isoformat()
                    if article.published_at is not None
                    else None,
                },
                id=stable_hash(article_id, str(index), chunk),
            )
            for index, chunk in enumerate(chunks)
        ]


class HashingEmbeddingModel:
    """A lightweight custom embedding model using signed feature hashing.

    It is deterministic, requires no pretrained weights, and is adequate for local retrieval
    tests. Production systems can swap this for an internally trained encoder while keeping
    the same `embed` interface.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts = Counter(token.lower().strip("$") for token in tokenize(text))
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign * (1.0 + math.log(count))
        return l2_normalize(vector)

    def embed_chunks(self, chunks: list[TextChunk]) -> list[TextChunk]:
        return [
            TextChunk(
                id=chunk.id,
                article_id=chunk.article_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                embedding=self.embed(chunk.text),
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]


class HybridRetriever:
    """Combines exact vector similarity search with graph neighborhood expansion."""

    def __init__(
        self,
        repository: object,
        embedding_model: HashingEmbeddingModel,
        entity_extractor: FinancialEntityExtractor,
        vector_top_k: int = 6,
        graph_neighbors_per_entity: int = 8,
    ) -> None:
        self.repository = repository
        self.embedding_model = embedding_model
        self.entity_extractor = entity_extractor
        self.vector_top_k = vector_top_k
        self.graph_neighbors_per_entity = graph_neighbors_per_entity

    def search(
        self,
        query: str,
        vector_top_k: int | None = None,
        graph_neighbors_per_entity: int | None = None,
        graph_hops: int = 1,
        include_chunks: bool = True,
        include_graph: bool = True,
    ) -> RetrievedContext:
        query_embedding = self.embedding_model.embed(query)
        chunks = (
            self.repository.search_chunks(query_embedding, limit=vector_top_k or self.vector_top_k)
            if include_chunks
            else []
        )

        query_entities = self.entity_extractor.extract_entities(query)
        if not query_entities:
            query_entities = self._find_nodes_from_terms(query)

        canonical_ids = [entity.canonical_id for entity in query_entities]
        neighbors = (
            self.repository.graph_neighbors(
                canonical_ids,
                limit_per_entity=graph_neighbors_per_entity or self.graph_neighbors_per_entity,
                hops=graph_hops,
            )
            if include_graph
            else []
        )
        return RetrievedContext(
            query=query,
            chunks=chunks,
            graph_neighbors=neighbors,
            query_entities=query_entities,
        )

    def _find_nodes_from_terms(self, query: str) -> list:
        candidates = []
        for token in tokenize(query):
            if len(token) < 3:
                continue
            candidates.extend(self.repository.find_nodes(token, limit=3))
        deduped = {}
        for candidate in candidates:
            deduped[candidate.canonical_id] = candidate
        return list(deduped.values())[: self.graph_neighbors_per_entity]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def tail_tokens(text: str, token_count: int) -> str:
    tokens = tokenize(text)
    return " ".join(tokens[-token_count:])


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
