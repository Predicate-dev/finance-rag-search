from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from graphrag_pipeline.demo_data import demo_finance_articles
from graphrag_pipeline.domain import Article
from graphrag_pipeline.interface.schemas import (
    ArticleIngestRequest,
    ArticleResult,
    IngestArticleResponse,
    PromptQueryRequest,
    SeedDemoResponse,
    dataclass_to_jsonable,
)
from graphrag_pipeline.interface.service import SearchGenerationService
from graphrag_pipeline.pipeline import GraphRAGPipeline


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on optional API extra.
        raise RuntimeError("Install the API extra with `pip install -e '.[api]'`.") from exc

    database_path = Path(os.environ.get("GRAPHRAG_DB_PATH", "graphrag.db"))
    checkpoint_path = os.environ.get("GRAPHRAG_MODEL_CHECKPOINT")
    pipeline = GraphRAGPipeline.local(
        database_path=database_path,
        model_checkpoint_path=checkpoint_path,
    )
    if os.environ.get("GRAPHRAG_SEED_DEMO", "true").lower() in {"1", "true", "yes"}:
        seed_demo_articles(pipeline, only_if_empty=True)

    service = SearchGenerationService(pipeline)
    app = FastAPI(
        title="GraphRAG Financial News API",
        version="0.1.0",
        description=(
            "Search and generate answers over a local financial-news GraphRAG index. "
            "Demo articles are seeded automatically unless GRAPHRAG_SEED_DEMO=false."
        ),
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/articles")
    def list_articles(limit: int = 50) -> dict[str, Any]:
        articles = pipeline.repository.list_articles(limit=limit)
        return {
            "count": len(articles),
            "articles": [
                dataclass_to_jsonable(
                    ArticleResult(
                        id=article.id,
                        source_url=article.source_url,
                        title=article.title,
                        body_preview=article.body[:300],
                        published_at=article.published_at,
                        source=article.source,
                    )
                )
                for article in articles
            ],
        }

    @app.post("/v1/articles")
    def ingest_article(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = ArticleIngestRequest.from_mapping(payload)
            article_id = pipeline.index_article(
                Article(
                    source_url=request.source_url,
                    title=request.title,
                    body=request.body,
                    published_at=request.published_at,
                    source=request.source,
                )
            )
            return IngestArticleResponse(article_id=article_id, indexed=True).to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/articles/bulk")
    def ingest_articles(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            article_payloads = payload.get("articles")
            if not isinstance(article_payloads, list):
                raise ValueError("Expected payload field 'articles' to be a list.")
            article_ids = []
            for item in article_payloads:
                request = ArticleIngestRequest.from_mapping(item)
                article_ids.append(
                    pipeline.index_article(
                        Article(
                            source_url=request.source_url,
                            title=request.title,
                            body=request.body,
                            published_at=request.published_at,
                            source=request.source,
                        )
                    )
                )
            return {"indexed_article_ids": article_ids, "total_articles": pipeline.repository.count_articles()}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/demo/seed")
    def seed_demo() -> dict[str, Any]:
        response = seed_demo_articles(pipeline, only_if_empty=False)
        return response.to_dict()

    @app.post("/v1/rag/query")
    def query(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = PromptQueryRequest.from_mapping(payload)
            response = service.search_and_generate(request=request)
            return response.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def seed_demo_articles(pipeline: GraphRAGPipeline, only_if_empty: bool) -> SeedDemoResponse:
    if only_if_empty and pipeline.repository.count_articles() > 0:
        return SeedDemoResponse(
            indexed_article_ids=[],
            total_articles=pipeline.repository.count_articles(),
        )
    article_ids = [pipeline.index_article(article) for article in demo_finance_articles()]
    return SeedDemoResponse(
        indexed_article_ids=article_ids,
        total_articles=pipeline.repository.count_articles(),
    )


try:
    app = create_app()
except RuntimeError:
    app = None
