# GraphRAG Financial News Pipeline

This project implements a modular GraphRAG pipeline for stock-market news:

1. Scrape and normalize financial articles from RSS feeds or mock HTML endpoints.
2. Store articles, chunks, graph nodes, graph edges, and vectors behind repository interfaces.
3. Extract financial entities, relations, and sentiment.
4. Build chunk embeddings and perform hybrid vector plus graph retrieval.
5. Train and run a decoder-only Transformer written from scratch in PyTorch.
6. Assemble retrieved context into a RAG prompt and generate an answer.

## Production Architecture

Recommended production storage is PostgreSQL with:

- `pgvector` for chunk embeddings and approximate nearest-neighbor search.
- Apache AGE for graph-style traversals inside PostgreSQL.
- Unique URL/content hashes for ingestion idempotency.

The package includes a SQLite repository for local development and tests. It stores embeddings as JSON and performs exact cosine search in Python. The same repository boundary can be replaced by a Postgres/AGE adapter without changing ingestion, extraction, retrieval, or generation code.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

## Run Everything Locally

1. Install dependencies:

```bash
cd /Users/shravanb/Programming/rag-pipeline
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

2. Run tests:

```bash
.venv/bin/python -m pytest
```

3. Run the local demo:

```bash
PYTHONPATH=src .venv/bin/python examples/local_demo.py
```

4. Train or retrain MiniGPT:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minigpt.py --epochs 80
```

5. Ask a question through the CLI:

```bash
.venv/bin/graphrag ask "How is Apple's supply chain looking after earnings?" \
  --db graphrag.db \
  --model artifacts/minigpt_finance/model.pt \
  --ticker AAPL \
  --template financial_rag \
  --top-k 8 \
  --graph-neighbors 10 \
  --debug \
  --json
```

6. Start the REST API:

```bash
GRAPHRAG_DB_PATH=graphrag.db \
GRAPHRAG_MODEL_CHECKPOINT=artifacts/minigpt_finance/model.pt \
.venv/bin/uvicorn graphrag_pipeline.interface.api:app --host 127.0.0.1 --port 8000
```

By default, the API seeds a demo financial-news corpus into an empty database. Set
`GRAPHRAG_SEED_DEMO=false` if you want to start with an empty index.

Then query it:

```bash
curl -X POST http://127.0.0.1:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How is Apple supply chain looking after earnings?",
    "template": "financial_rag",
    "filters": {"tickers": ["AAPL"]},
    "retrieval": {"vector_top_k": 8, "graph_neighbors_per_entity": 10},
    "generation": {"max_tokens": 120, "temperature": 0.25},
    "debug": true
  }'
```

List indexed articles:

```bash
curl http://127.0.0.1:8000/v1/articles
```

Add your own article:

```bash
curl -X POST http://127.0.0.1:8000/v1/articles \
  -H "Content-Type: application/json" \
  -d '{
    "source_url": "api://oracle-cloud-margin",
    "title": "Oracle raises cloud guidance as margins improve",
    "body": "Oracle raised cloud guidance after strong enterprise demand and improved margins.",
    "published_at": "2026-05-20T12:00:00Z",
    "source": "api"
  }'
```

Force-load the demo corpus again:

```bash
curl -X POST http://127.0.0.1:8000/v1/demo/seed
```

Minimal local run:

```python
from datetime import datetime, timezone

from graphrag_pipeline.domain import Article
from graphrag_pipeline.pipeline import GraphRAGPipeline

pipeline = GraphRAGPipeline.local("graphrag.db", initialize_model=False)
article = Article(
    source_url="mock://aapl-earnings",
    title="Apple beats estimates as services revenue grows",
    body="Apple beat analyst estimates. CEO Tim Cook said supply chain pressure eased.",
    published_at=datetime.now(timezone.utc),
    source="mock",
)
pipeline.index_article(article)
answer = pipeline.answer("How is Apple's supply chain looking after earnings?")
print(answer["response"])
```

For actual generation, train the custom Transformer first with `train_language_model` in `src/graphrag_pipeline/transformer.py`, then pass the model and tokenizer into `GraphRAGPipeline`.

## Train MiniGPT

```bash
PYTHONPATH=src .venv/bin/python scripts/train_minigpt.py --epochs 12
```

Use the trained checkpoint with the pipeline:

```python
from graphrag_pipeline.pipeline import GraphRAGPipeline

pipeline = GraphRAGPipeline.local(
    "graphrag.db",
    model_checkpoint_path="artifacts/minigpt_finance/model.pt",
)
```

## Prompt/Search Interface

Python SDK:

```python
from graphrag_pipeline.interface import SearchGenerationService
from graphrag_pipeline.pipeline import GraphRAGPipeline

pipeline = GraphRAGPipeline.local(
    "graphrag.db",
    model_checkpoint_path="artifacts/minigpt_finance/model.pt",
)
client = SearchGenerationService(pipeline)

result = client.search_and_generate(
    query="How is Apple's supply chain looking after earnings?",
    template="financial_rag",
    filters={"tickers": ["AAPL"], "sources": ["mock"]},
    retrieval={"vector_top_k": 8, "graph_neighbors_per_entity": 10},
    generation={"max_tokens": 120, "temperature": 0.25, "top_k": 8},
    debug=True,
)
print(result.to_dict())
```

CLI:

```bash
PYTHONPATH=src .venv/bin/python -m graphrag_pipeline.interface.cli ask \
  "How is Apple's supply chain looking after earnings?" \
  --db graphrag.db \
  --model artifacts/minigpt_finance/model.pt \
  --ticker AAPL \
  --template financial_rag \
  --top-k 8 \
  --debug \
  --json
```

REST API:

```bash
GRAPHRAG_DB_PATH=graphrag.db \
GRAPHRAG_MODEL_CHECKPOINT=artifacts/minigpt_finance/model.pt \
uvicorn graphrag_pipeline.interface.api:app --host 127.0.0.1 --port 8000
```

```http
POST /v1/rag/query
```

```json
{
  "query": "How is Apple's supply chain looking after earnings?",
  "template": "financial_rag",
  "filters": {"tickers": ["AAPL"]},
  "retrieval": {"vector_top_k": 8, "include_graph": true},
  "generation": {"max_tokens": 120, "temperature": 0.25},
  "debug": true
}
```
