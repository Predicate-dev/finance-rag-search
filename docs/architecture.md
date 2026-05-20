# GraphRAG Pipeline Architecture

## Core Design

The system is organized around stable object boundaries:

- `Article` and `TextChunk` domain objects move data between modules.
- `SQLiteGraphRAGRepository` provides the local article, vector, node, and edge store.
- Production storage should use PostgreSQL with `pgvector` and Apache AGE, described in `src/graphrag_pipeline/schema/postgres_apache_age.sql`.
- The LLM path uses a custom decoder-only Transformer implemented from scratch in PyTorch. It does not load pretrained Hugging Face weights.

## Module 1: Web Scraper & Ingestion

`FinancialNewsIngestionEngine` accepts `NewsSource` definitions for RSS or HTML pages. It fetches sources with request headers, rate limiting, timeouts, error logging, RSS parsing, article-page extraction, and HTML cleanup.

## Module 2: Database & Storage

`SQLiteGraphRAGRepository` stores:

- raw articles with unique source URLs,
- text chunks and embeddings,
- graph nodes for entities,
- graph edges for relations and sentiment-bearing evidence.

The production SQL schema includes `pgvector` HNSW indexing for vectors and Apache AGE initialization for graph traversal.

## Module 3: Entity, Sentiment, and Graph Construction

`FinancialEntityExtractor` extracts companies, tickers, people, and financial event nodes using transparent finance-oriented rules. `FinancialSentimentAnalyzer` assigns scores from `-1.0` to `1.0`. `KnowledgeGraphBuilder` upserts extracted entities and relations into the graph store.

## Module 4: Vectorization & Hybrid Retrieval

`FinancialTextChunker` chunks article text by sentence with overlap. `HashingEmbeddingModel` creates deterministic custom embeddings without pretrained weights. `HybridRetriever` combines vector similarity with graph-neighbor expansion around query entities.

## Module 5: Custom Transformer

`MiniGPT` implements:

- token and positional embeddings,
- causal multi-head self-attention,
- decoder blocks with LayerNorm and feed-forward layers,
- tied output head,
- autoregressive generation,
- a basic next-token training loop.

`FinancialTokenizer` handles finance-specific tokens, tickers, percentages, and RAG section markers.

## Module 6: RAG Execution

`GraphRAGPipeline` orchestrates ingestion, indexing, retrieval, prompt formatting, generation, and sentiment breakdown. If no trained model is passed, it uses a deterministic extractive fallback for local smoke tests; production generation should pass a trained `MiniGPT` instance.
