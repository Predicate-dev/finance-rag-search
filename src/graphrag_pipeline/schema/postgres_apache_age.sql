CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

SELECT create_graph('financial_news_graph')
WHERE NOT EXISTS (
  SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'financial_news_graph'
);

CREATE TABLE IF NOT EXISTS articles (
  id UUID PRIMARY KEY,
  source_url TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  published_at TIMESTAMPTZ,
  source TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id UUID PRIMARY KEY,
  article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  embedding VECTOR(384) NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(article_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS entities (
  canonical_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  sentiment DOUBLE PRECISION NOT NULL DEFAULT 0,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS relations (
  id UUID PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES entities(canonical_id) ON DELETE CASCADE,
  target_id TEXT NOT NULL REFERENCES entities(canonical_id) ON DELETE CASCADE,
  relation_type TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  sentiment DOUBLE PRECISION NOT NULL,
  evidence TEXT NOT NULL,
  article_id UUID REFERENCES articles(id) ON DELETE SET NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS relations_source_idx ON relations(source_id);
CREATE INDEX IF NOT EXISTS relations_target_idx ON relations(target_id);
