from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from graphrag_pipeline.interface.schemas import PromptQueryRequest
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
    service = SearchGenerationService(pipeline)
    app = FastAPI(title="GraphRAG Financial News API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/rag/query")
    def query(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = PromptQueryRequest.from_mapping(payload)
            response = service.search_and_generate(request=request)
            return response.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


try:
    app = create_app()
except RuntimeError:
    app = None
