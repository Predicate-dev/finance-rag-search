"""GraphRAG pipeline for financial news."""

from graphrag_pipeline.config import PipelineConfig
from graphrag_pipeline.domain import Article
from graphrag_pipeline.pipeline import GraphRAGPipeline

__all__ = ["Article", "GraphRAGPipeline", "PipelineConfig"]
