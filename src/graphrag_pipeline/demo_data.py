from __future__ import annotations

from datetime import datetime, timezone

from graphrag_pipeline.domain import Article


def demo_finance_articles() -> list[Article]:
    """A small but substantial searchable corpus for local API demos."""

    published = datetime(2026, 5, 20, 13, 30, tzinfo=timezone.utc)
    return [
        Article(
            source_url="demo://apple-supply-chain-earnings",
            title="Apple beats estimates as supply chain pressure eases",
            body=(
                "Apple beat analyst estimates after services revenue grew faster than expected. "
                "CEO Tim Cook said supply chain pressure eased during the quarter, with component "
                "availability improving across iPhone and Mac product lines. Analysts remained "
                "bullish because gross margins improved, channel inventory normalized, and demand "
                "in India stayed resilient. The company also raised its services outlook."
            ),
            published_at=published,
            source="demo",
        ),
        Article(
            source_url="demo://nvidia-data-center-orders",
            title="Nvidia rallies on record data center revenue and strong AI orders",
            body=(
                "Nvidia shares rallied after data center revenue reached a record. Management said "
                "supply constraints were improving and hyperscaler orders remained strong. Analysts "
                "upgraded the stock as AI accelerator demand continued to exceed supply, although "
                "some warned that export controls could create regional revenue risk."
            ),
            published_at=published,
            source="demo",
        ),
        Article(
            source_url="demo://microsoft-cloud-ai-guidance",
            title="Microsoft raises guidance as Azure and enterprise AI demand accelerate",
            body=(
                "Microsoft reported strong cloud growth and raised guidance. Azure revenue "
                "accelerated as enterprise AI demand remained resilient. CFO commentary pointed to "
                "continued capital spending on AI infrastructure, but analysts viewed the update as "
                "bullish because backlog growth and margin discipline offset spending concerns."
            ),
            published_at=published,
            source="demo",
        ),
        Article(
            source_url="demo://tesla-demand-margin-pressure",
            title="Tesla misses delivery estimates as price pressure weighs on margins",
            body=(
                "Tesla missed delivery estimates after weaker demand in Europe and rising price "
                "pressure. Analysts downgraded the stock and warned that automotive margins could "
                "decline further if incentives remain elevated. Management said new product launches "
                "may improve demand later in the year, but the near-term sentiment stayed bearish."
            ),
            published_at=published,
            source="demo",
        ),
        Article(
            source_url="demo://amd-ai-chip-forecast",
            title="AMD raises AI chip forecast after cloud customer orders strengthen",
            body=(
                "AMD raised its AI chip revenue forecast after strong orders from cloud customers. "
                "Gross margin guidance improved and analysts upgraded the stock. Management said "
                "supply chain execution had improved, allowing the company to ship more accelerators "
                "into enterprise and hyperscaler deployments."
            ),
            published_at=published,
            source="demo",
        ),
        Article(
            source_url="demo://bank-credit-risk",
            title="Bank stocks mixed as deposit costs and credit losses remain in focus",
            body=(
                "JPMorgan beat profit estimates as net interest income stayed strong. Bank of "
                "America warned that deposit costs remained elevated and loan growth slowed. "
                "Executives across the sector said credit losses could rise if unemployment "
                "increases, creating a mixed sentiment backdrop for large banks."
            ),
            published_at=published,
            source="demo",
        ),
    ]
