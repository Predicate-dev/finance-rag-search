from __future__ import annotations

from datetime import datetime, timezone

from graphrag_pipeline.domain import Article
from graphrag_pipeline.pipeline import GraphRAGPipeline


def main() -> None:
    pipeline = GraphRAGPipeline.local("demo_graphrag.db", initialize_model=False)
    pipeline.index_article(
        Article(
            source_url="mock://apple-earnings",
            title="Apple beats estimates as services growth offsets supply chain pressure",
            body=(
                "Apple beat analyst estimates after strong services revenue. "
                "CEO Tim Cook said supply chain pressure eased during the quarter. "
                "Analysts remained bullish as margins improved and iPhone demand stayed resilient."
            ),
            published_at=datetime.now(timezone.utc),
            source="mock",
        )
    )
    result = pipeline.answer("How is Apple's supply chain looking after the latest earnings?")
    print(result["response"])
    print(result["sentiment"])


if __name__ == "__main__":
    main()
