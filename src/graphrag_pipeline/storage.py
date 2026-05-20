from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graphrag_pipeline.domain import Article, Entity, GraphNeighbor, Relation, RetrievedChunk, TextChunk


def stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="ignore"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _dt_to_str(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _str_to_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


class SQLiteGraphRAGRepository(AbstractContextManager["SQLiteGraphRAGRepository"]):
    """Local repository implementing article, graph, and vector persistence.

    Production deployments should use the SQL in schema/postgres_apache_age.sql and provide
    the same method surface with pgvector-backed similarity search.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    def __exit__(self, *args: object) -> None:
        self.connection.close()

    def initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                source_url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                published_at TEXT,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                article_id TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(article_id, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS nodes (
                canonical_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                sentiment REAL NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES nodes(canonical_id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES nodes(canonical_id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                sentiment REAL NOT NULL,
                evidence TEXT NOT NULL,
                article_id TEXT REFERENCES articles(id) ON DELETE SET NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            """
        )
        self.connection.commit()

    def upsert_article(self, article: Article) -> str:
        article_id = article.id or stable_hash(article.source_url)
        self.connection.execute(
            """
            INSERT INTO articles(id, source_url, title, body, published_at, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_url) DO UPDATE SET
                title = excluded.title,
                body = excluded.body,
                published_at = excluded.published_at,
                source = excluded.source
            """,
            (
                article_id,
                article.source_url,
                article.title,
                article.body,
                _dt_to_str(article.published_at),
                article.source,
                _dt_to_str(article.created_at),
            ),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT id FROM articles WHERE source_url = ?", (article.source_url,)
        ).fetchone()
        return str(row["id"])

    def get_article(self, article_id: str) -> Article | None:
        row = self.connection.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        if row is None:
            return None
        return Article(
            id=row["id"],
            source_url=row["source_url"],
            title=row["title"],
            body=row["body"],
            published_at=_str_to_dt(row["published_at"]),
            source=row["source"],
            created_at=_str_to_dt(row["created_at"]) or datetime.now(timezone.utc),
        )

    def count_articles(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM articles").fetchone()
        return int(row["count"])

    def list_articles(self, limit: int = 50) -> list[Article]:
        rows = self.connection.execute(
            """
            SELECT * FROM articles
            ORDER BY COALESCE(published_at, created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            Article(
                id=row["id"],
                source_url=row["source_url"],
                title=row["title"],
                body=row["body"],
                published_at=_str_to_dt(row["published_at"]),
                source=row["source"],
                created_at=_str_to_dt(row["created_at"]) or datetime.now(timezone.utc),
            )
            for row in rows
        ]

    def upsert_chunks(self, chunks: Iterable[TextChunk]) -> None:
        rows = []
        for chunk in chunks:
            chunk_id = chunk.id or stable_hash(chunk.article_id, str(chunk.chunk_index), chunk.text)
            rows.append(
                (
                    chunk_id,
                    chunk.article_id,
                    chunk.chunk_index,
                    chunk.text,
                    json.dumps(chunk.embedding or []),
                    json.dumps(chunk.metadata, sort_keys=True),
                )
            )
        self.connection.executemany(
            """
            INSERT INTO chunks(id, article_id, chunk_index, text, embedding_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(article_id, chunk_index) DO UPDATE SET
                text = excluded.text,
                embedding_json = excluded.embedding_json,
                metadata_json = excluded.metadata_json
            """,
            rows,
        )
        self.connection.commit()

    def upsert_entity(self, entity: Entity) -> str:
        self.connection.execute(
            """
            INSERT INTO nodes(canonical_id, name, entity_type, sentiment, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(canonical_id) DO UPDATE SET
                name = excluded.name,
                entity_type = excluded.entity_type,
                sentiment = (nodes.sentiment + excluded.sentiment) / 2.0,
                metadata_json = excluded.metadata_json
            """,
            (
                entity.canonical_id,
                entity.name,
                entity.entity_type,
                entity.sentiment,
                json.dumps(entity.metadata, sort_keys=True),
            ),
        )
        self.connection.commit()
        return entity.canonical_id

    def upsert_relation(self, relation: Relation) -> str:
        edge_id = stable_hash(
            relation.source.canonical_id,
            relation.target.canonical_id,
            relation.relation_type,
            relation.article_id or "",
            relation.evidence[:240],
        )
        self.connection.execute(
            """
            INSERT INTO edges(
                id, source_id, target_id, relation_type, confidence,
                sentiment, evidence, article_id, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                confidence = excluded.confidence,
                sentiment = excluded.sentiment,
                evidence = excluded.evidence,
                metadata_json = excluded.metadata_json
            """,
            (
                edge_id,
                relation.source.canonical_id,
                relation.target.canonical_id,
                relation.relation_type,
                relation.confidence,
                relation.sentiment,
                relation.evidence,
                relation.article_id,
                json.dumps(relation.metadata, sort_keys=True),
            ),
        )
        self.connection.commit()
        return edge_id

    def find_nodes(self, query: str, limit: int = 8) -> list[Entity]:
        like = f"%{query.lower()}%"
        rows = self.connection.execute(
            """
            SELECT * FROM nodes
            WHERE lower(name) LIKE ? OR lower(canonical_id) LIKE ?
            ORDER BY entity_type, name
            LIMIT ?
            """,
            (like, like, limit),
        ).fetchall()
        return [
            Entity(
                name=row["name"],
                entity_type=row["entity_type"],
                canonical_id=row["canonical_id"],
                sentiment=float(row["sentiment"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            for row in rows
        ]

    def graph_neighbors(
        self,
        canonical_ids: list[str],
        limit_per_entity: int = 8,
        hops: int = 1,
    ) -> list[GraphNeighbor]:
        if not canonical_ids:
            return []
        neighbors: list[GraphNeighbor] = []
        seen_edges: set[str] = set()
        frontier = set(canonical_ids)
        visited_nodes = set(canonical_ids)

        for _ in range(max(1, hops)):
            next_frontier: set[str] = set()
            if not frontier:
                break
            for canonical_id in sorted(frontier):
                rows = self.connection.execute(
                    """
                    SELECT e.id, e.source_id, e.target_id,
                           sn.name AS source_name, tn.name AS target_name, e.relation_type,
                           e.sentiment, e.evidence
                    FROM edges e
                    JOIN nodes sn ON sn.canonical_id = e.source_id
                    JOIN nodes tn ON tn.canonical_id = e.target_id
                    WHERE e.source_id = ? OR e.target_id = ?
                    ORDER BY abs(e.sentiment) DESC, e.confidence DESC
                    LIMIT ?
                    """,
                    (canonical_id, canonical_id, limit_per_entity),
                ).fetchall()
                for row in rows:
                    if row["id"] not in seen_edges:
                        seen_edges.add(row["id"])
                        neighbors.append(
                            GraphNeighbor(
                                source_name=row["source_name"],
                                relation_type=row["relation_type"],
                                target_name=row["target_name"],
                                sentiment=float(row["sentiment"]),
                                evidence=row["evidence"],
                            )
                        )
                    for node_id in (row["source_id"], row["target_id"]):
                        if node_id not in visited_nodes:
                            visited_nodes.add(node_id)
                            next_frontier.add(node_id)
            frontier = next_frontier

        return neighbors

    def search_chunks(self, query_embedding: list[float], limit: int = 6) -> list[RetrievedChunk]:
        rows = self.connection.execute(
            """
            SELECT c.*, a.title, a.source_url
            FROM chunks c
            JOIN articles a ON a.id = c.article_id
            """
        ).fetchall()
        scored: list[RetrievedChunk] = []
        for row in rows:
            embedding = json.loads(row["embedding_json"])
            score = cosine_similarity(query_embedding, embedding)
            scored.append(
                RetrievedChunk(
                    article_id=row["article_id"],
                    title=row["title"],
                    source_url=row["source_url"],
                    text=row["text"],
                    score=score,
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
