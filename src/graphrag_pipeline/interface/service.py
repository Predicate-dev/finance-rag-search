from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from graphrag_pipeline.domain import RetrievedChunk, RetrievedContext
from graphrag_pipeline.entities import FinancialEntityExtractor
from graphrag_pipeline.interface.prompt_templates import (
    DEFAULT_PROMPT_TEMPLATES,
    PromptTemplateRegistry,
)
from graphrag_pipeline.interface.schemas import (
    ChunkResult,
    EntityResult,
    GenerationOptions,
    GraphNeighborResult,
    PromptQueryRequest,
    PromptResult,
    RetrievalOptions,
    RetrievalResult,
    SearchFilters,
    SearchGenerationResponse,
    SentimentResult,
)
from graphrag_pipeline.pipeline import GraphRAGPipeline


class SearchGenerationService:
    """SDK facade for prompt injection, hybrid search controls, and generation."""

    def __init__(
        self,
        pipeline: GraphRAGPipeline,
        templates: PromptTemplateRegistry | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.templates = templates or DEFAULT_PROMPT_TEMPLATES

    def search_and_generate(
        self,
        query: str | None = None,
        template: str = "financial_rag",
        filters: SearchFilters | dict[str, Any] | None = None,
        retrieval: RetrievalOptions | dict[str, Any] | None = None,
        generation: GenerationOptions | dict[str, Any] | None = None,
        debug: bool = False,
        request: PromptQueryRequest | dict[str, Any] | None = None,
    ) -> SearchGenerationResponse:
        request_obj = self._coerce_request(
            query=query,
            template=template,
            filters=filters,
            retrieval=retrieval,
            generation=generation,
            debug=debug,
            request=request,
        )
        enhanced_query = self._enhance_query(request_obj.query, request_obj.filters)
        context = self.pipeline.retrieve(
            query=enhanced_query,
            vector_top_k=request_obj.retrieval.vector_top_k,
            graph_neighbors_per_entity=request_obj.retrieval.graph_neighbors_per_entity,
            graph_hops=request_obj.retrieval.graph_hops,
            include_chunks=request_obj.retrieval.include_chunks,
            include_graph=request_obj.retrieval.include_graph,
        )
        context = replace(
            context,
            query=request_obj.query,
            chunks=self._filter_chunks(context.chunks, request_obj.filters),
            graph_neighbors=self._filter_graph_neighbors(
                context.graph_neighbors, request_obj.filters
            ),
        )

        prompt = self.templates.get(request_obj.template).render(context)
        answer = self.pipeline.generate(
            prompt=prompt,
            max_tokens=request_obj.generation.max_tokens,
            temperature=request_obj.generation.temperature,
            top_k=request_obj.generation.top_k,
        )
        sentiment = (
            self._sentiment_result(request_obj.query, context)
            if request_obj.retrieval.include_sentiment
            else None
        )
        return SearchGenerationResponse(
            answer=answer,
            sentiment=sentiment,
            retrieval=self._retrieval_result(context),
            prompt=PromptResult(
                template=request_obj.template,
                rendered=prompt if request_obj.debug else None,
            ),
        )

    def _coerce_request(
        self,
        query: str | None,
        template: str,
        filters: SearchFilters | dict[str, Any] | None,
        retrieval: RetrievalOptions | dict[str, Any] | None,
        generation: GenerationOptions | dict[str, Any] | None,
        debug: bool,
        request: PromptQueryRequest | dict[str, Any] | None,
    ) -> PromptQueryRequest:
        if isinstance(request, PromptQueryRequest):
            return request
        if isinstance(request, dict):
            return PromptQueryRequest.from_mapping(request)
        if query is None:
            raise ValueError("query is required")
        return PromptQueryRequest(
            query=query,
            template=template,
            filters=filters
            if isinstance(filters, SearchFilters)
            else SearchFilters.from_mapping(filters),
            retrieval=retrieval
            if isinstance(retrieval, RetrievalOptions)
            else RetrievalOptions.from_mapping(retrieval),
            generation=generation
            if isinstance(generation, GenerationOptions)
            else GenerationOptions.from_mapping(generation),
            debug=debug,
        )

    @staticmethod
    def _enhance_query(query: str, filters: SearchFilters) -> str:
        additions = [*filters.tickers, *filters.entities]
        if not additions:
            return query
        return " ".join([query, *additions])

    def _filter_chunks(
        self, chunks: list[RetrievedChunk], filters: SearchFilters
    ) -> list[RetrievedChunk]:
        return [chunk for chunk in chunks if self._chunk_matches(chunk, filters)]

    def _chunk_matches(self, chunk: RetrievedChunk, filters: SearchFilters) -> bool:
        metadata = chunk.metadata or {}
        if filters.sources:
            source = str(metadata.get("source") or "").lower()
            if source not in {item.lower() for item in filters.sources}:
                return False

        published_at = parse_metadata_datetime(metadata.get("published_at"))
        if filters.published_after and published_at and published_at < filters.published_after:
            return False
        if filters.published_before and published_at and published_at > filters.published_before:
            return False

        text_blob = " ".join([chunk.title, chunk.source_url, chunk.text]).lower()
        ticker_terms = expand_ticker_terms(filters.tickers)
        if ticker_terms and not any(term in text_blob for term in ticker_terms):
            return False
        if filters.entities and not any(entity.lower() in text_blob for entity in filters.entities):
            return False
        return True

    @staticmethod
    def _filter_graph_neighbors(graph_neighbors: list, filters: SearchFilters) -> list:
        if filters.min_sentiment is None and filters.max_sentiment is None:
            return graph_neighbors
        filtered = []
        for edge in graph_neighbors:
            if filters.min_sentiment is not None and edge.sentiment < filters.min_sentiment:
                continue
            if filters.max_sentiment is not None and edge.sentiment > filters.max_sentiment:
                continue
            filtered.append(edge)
        return filtered

    def _sentiment_result(self, query: str, context: RetrievedContext) -> SentimentResult:
        text = " ".join([query, *[chunk.text for chunk in context.chunks]])
        breakdown = self.pipeline.entity_extractor.sentiment_breakdown(text, [])
        return SentimentResult(
            overall=breakdown.overall,
            entities=breakdown.by_entity,
            relations={
                f"{edge.source_name} {edge.relation_type} {edge.target_name}": edge.sentiment
                for edge in context.graph_neighbors
            },
        )

    @staticmethod
    def _retrieval_result(context: RetrievedContext) -> RetrievalResult:
        return RetrievalResult(
            chunks=[
                ChunkResult(
                    article_id=chunk.article_id,
                    title=chunk.title,
                    source_url=chunk.source_url,
                    text=chunk.text,
                    score=chunk.score,
                    metadata=chunk.metadata,
                )
                for chunk in context.chunks
            ],
            graph_neighbors=[
                GraphNeighborResult(
                    source_name=edge.source_name,
                    relation_type=edge.relation_type,
                    target_name=edge.target_name,
                    sentiment=edge.sentiment,
                    evidence=edge.evidence,
                )
                for edge in context.graph_neighbors
            ],
            query_entities=[
                EntityResult(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    canonical_id=entity.canonical_id,
                    sentiment=entity.sentiment,
                    confidence=entity.confidence,
                )
                for entity in context.query_entities
            ],
        )


def parse_metadata_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def expand_ticker_terms(tickers: list[str]) -> set[str]:
    terms = {ticker.lower() for ticker in tickers}
    known = FinancialEntityExtractor.KNOWN_COMPANIES
    for company, ticker in known.items():
        if ticker.lower() in terms or company.lower() in terms:
            terms.add(ticker.lower())
            terms.add(company.lower())
    return terms
