from __future__ import annotations

from pathlib import Path
from typing import Any

from graphrag_pipeline.config import PipelineConfig
from graphrag_pipeline.domain import Article, RetrievedContext
from graphrag_pipeline.entities import (
    FinancialEntityExtractor,
    FinancialSentimentAnalyzer,
    KnowledgeGraphBuilder,
)
from graphrag_pipeline.ingestion import FinancialNewsIngestionEngine, NewsSource
from graphrag_pipeline.storage import SQLiteGraphRAGRepository
from graphrag_pipeline.tokenizer import FinancialTokenizer
from graphrag_pipeline.vectorization import FinancialTextChunker, HashingEmbeddingModel, HybridRetriever


class GraphRAGPipeline:
    """End-to-end orchestrator for ingestion, graph indexing, retrieval, and generation."""

    def __init__(
        self,
        config: PipelineConfig,
        repository: SQLiteGraphRAGRepository,
        ingestion_engine: FinancialNewsIngestionEngine,
        entity_extractor: FinancialEntityExtractor,
        graph_builder: KnowledgeGraphBuilder,
        chunker: FinancialTextChunker,
        embedding_model: HashingEmbeddingModel,
        tokenizer: FinancialTokenizer | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.ingestion_engine = ingestion_engine
        self.entity_extractor = entity_extractor
        self.graph_builder = graph_builder
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.model = model
        self.retriever = HybridRetriever(
            repository=repository,
            embedding_model=embedding_model,
            entity_extractor=entity_extractor,
            vector_top_k=config.vector_top_k,
            graph_neighbors_per_entity=config.graph_neighbors_per_entity,
        )

    @classmethod
    def local(
        cls,
        database_path: str | Path = "graphrag.db",
        initialize_model: bool = False,
        model_checkpoint_path: str | Path | None = None,
    ) -> "GraphRAGPipeline":
        config = PipelineConfig(database_path=Path(database_path))
        repository = SQLiteGraphRAGRepository(config.database_path)
        repository.initialize_schema()
        sentiment = FinancialSentimentAnalyzer()
        extractor = FinancialEntityExtractor(sentiment)
        tokenizer = FinancialTokenizer()
        model = None

        if model_checkpoint_path is not None:
            from graphrag_pipeline.transformer import load_minigpt_checkpoint

            model, tokenizer = load_minigpt_checkpoint(model_checkpoint_path)
        elif initialize_model:
            from graphrag_pipeline.transformer import MiniGPTConfig, build_model

            model = build_model(
                MiniGPTConfig(
                    vocab_size=tokenizer.vocab_size,
                    max_seq_len=config.max_prompt_tokens,
                )
            )

        return cls(
            config=config,
            repository=repository,
            ingestion_engine=FinancialNewsIngestionEngine(
                requests_per_second=config.requests_per_second,
                timeout_seconds=config.request_timeout_seconds,
            ),
            entity_extractor=extractor,
            graph_builder=KnowledgeGraphBuilder(extractor),
            chunker=FinancialTextChunker(config.chunk_token_budget, config.chunk_overlap_tokens),
            embedding_model=HashingEmbeddingModel(config.embedding_dim),
            tokenizer=tokenizer,
            model=model,
        )

    def ingest_sources(
        self, sources: list[NewsSource], max_articles_per_source: int = 25
    ) -> list[str]:
        article_ids: list[str] = []
        for article in self.ingestion_engine.scrape(sources, max_articles_per_source):
            article_ids.append(self.index_article(article))
        return article_ids

    def index_article(self, article: Article) -> str:
        article_id = self.repository.upsert_article(article)
        chunks = self.chunker.chunk_article(article, article_id)
        embedded_chunks = self.embedding_model.embed_chunks(chunks)
        self.repository.upsert_chunks(embedded_chunks)
        self.graph_builder.build_for_article(article, article_id, self.repository)
        return article_id

    def answer(self, query: str) -> dict[str, Any]:
        context = self.retrieve(query)
        prompt = self.format_prompt(context)
        sentiment = self.entity_extractor.sentiment_breakdown(
            " ".join([query, *[chunk.text for chunk in context.chunks]]),
            [],
        )
        response = self.generate(prompt)
        return {
            "query": query,
            "response": response,
            "prompt": prompt,
            "retrieved_chunks": context.chunks,
            "graph_neighbors": context.graph_neighbors,
            "sentiment": sentiment,
        }

    def retrieve(
        self,
        query: str,
        vector_top_k: int | None = None,
        graph_neighbors_per_entity: int | None = None,
        graph_hops: int = 1,
        include_chunks: bool = True,
        include_graph: bool = True,
    ) -> RetrievedContext:
        return self.retriever.search(
            query=query,
            vector_top_k=vector_top_k,
            graph_neighbors_per_entity=graph_neighbors_per_entity,
            graph_hops=graph_hops,
            include_chunks=include_chunks,
            include_graph=include_graph,
        )

    def format_prompt(self, context: RetrievedContext) -> str:
        graph_lines = [
            f"- {edge.source_name} {edge.relation_type} {edge.target_name} "
            f"(sentiment={edge.sentiment:.2f})"
            for edge in context.graph_neighbors
        ]
        chunk_lines = [
            chunk.text for chunk in context.chunks
        ]
        return (
            "<CONTEXT>\n"
            + "\n\n".join(chunk_lines)
            + "\n\n<GRAPH>\n"
            + ("\n".join(graph_lines) if graph_lines else "- No graph neighbors found.")
            + "\n\n<QUESTION>\n"
            + context.query
            + "\n\n<ANSWER>\n"
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> str:
        return self._generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    def _generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            return self._extractive_response(prompt)

        try:
            import torch
        except ImportError:
            return self._extractive_response(prompt)

        self.model.eval()
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = prompt_ids[-(self.config.max_prompt_tokens - 1) :]
        input_ids = [self.tokenizer.bos_id, *prompt_ids]
        device = next(self.model.parameters()).device
        tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        generated = self.model.generate(
            tensor,
            max_new_tokens=max_tokens or self.config.generation_tokens,
            temperature=temperature
            if temperature is not None
            else self.config.generation_temperature,
            top_k=top_k if top_k is not None else self.config.generation_top_k,
            eos_id=self.tokenizer.eos_id,
        )[0].detach().cpu().tolist()
        return self.tokenizer.decode(generated[len(input_ids) :]).strip()

    @staticmethod
    def _extractive_response(prompt: str) -> str:
        """Deterministic fallback for untrained local runs."""

        context = extract_prompt_section(prompt, "CONTEXT", "GRAPH")
        first_context = context.split("\n\n", maxsplit=1)[0].strip()
        if not first_context or first_context.startswith("- No text chunks retrieved"):
            return "I do not have enough retrieved market context to answer confidently."
        return (
            "Based on the retrieved financial news context, the most relevant evidence is: "
            f"{first_context[:700]}"
        )


def extract_prompt_section(prompt: str, start_tag: str, end_tag: str) -> str:
    start = f"<{start_tag}>"
    end = f"<{end_tag}>"
    if start not in prompt:
        return ""
    section = prompt.split(start, maxsplit=1)[1]
    if end in section:
        section = section.split(end, maxsplit=1)[0]
    return section.strip()
