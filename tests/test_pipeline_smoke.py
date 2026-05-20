from __future__ import annotations

from datetime import datetime, timezone

from graphrag_pipeline.domain import Article
from graphrag_pipeline.entities import FinancialSentimentAnalyzer
from graphrag_pipeline.pipeline import GraphRAGPipeline


def test_sentiment_scores_financial_terms() -> None:
    analyzer = FinancialSentimentAnalyzer()
    assert analyzer.score("Apple beat estimates and raised guidance") > 0
    assert analyzer.score("Tesla missed estimates after weak demand") < 0


def test_local_pipeline_indexes_and_retrieves(tmp_path) -> None:
    pipeline = GraphRAGPipeline.local(tmp_path / "test.db", initialize_model=False)
    pipeline.index_article(
        Article(
            source_url="mock://aapl",
            title="Apple beats estimates as supply chain pressure eased",
            body=(
                "Apple Inc beat analyst estimates. CEO Tim Cook said supply chain pressure eased. "
                "The company remained bullish on services growth."
            ),
            published_at=datetime.now(timezone.utc),
            source="mock",
        )
    )
    result = pipeline.answer("How is Apple's supply chain looking?")
    assert result["retrieved_chunks"]
    assert "retrieved financial news context" in result["response"]
