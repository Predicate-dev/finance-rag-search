from __future__ import annotations

import html
import logging
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable

from graphrag_pipeline.domain import Article

try:  # BeautifulSoup is preferred for HTML article parsing.
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - dependency is declared, fallback keeps module importable.
    BeautifulSoup = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsSource:
    """A scrape target. RSS sources work well for finance news and mock endpoints."""

    name: str
    url: str
    kind: str = "rss"  # "rss" or "html"
    article_selector: str | None = None


class RateLimiter:
    """Small token-free limiter suitable for polite RSS and article fetching."""

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._min_interval = 1.0 / requests_per_second
        self._last_request_at = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


class ArticleCleaner:
    """Normalizes HTML and text artifacts commonly found in scraped articles."""

    BOILERPLATE_PATTERNS = [
        re.compile(r"subscribe now", re.IGNORECASE),
        re.compile(r"sign up for.*newsletter", re.IGNORECASE),
        re.compile(r"advertisement", re.IGNORECASE),
        re.compile(r"all rights reserved", re.IGNORECASE),
    ]

    def clean_text(self, value: str) -> str:
        value = html.unescape(value or "")
        value = re.sub(r"<[^>]+>", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        for pattern in self.BOILERPLATE_PATTERNS:
            value = pattern.sub("", value)
        return re.sub(r"\s+", " ", value).strip()


class FinancialNewsIngestionEngine:
    """Scrapes finance news feeds and article pages into canonical Article objects."""

    def __init__(
        self,
        requests_per_second: float = 0.5,
        timeout_seconds: float = 10.0,
        cleaner: ArticleCleaner | None = None,
    ) -> None:
        self.rate_limiter = RateLimiter(requests_per_second)
        self.timeout_seconds = timeout_seconds
        self.cleaner = cleaner or ArticleCleaner()

    def scrape(self, sources: Iterable[NewsSource], max_articles_per_source: int = 25) -> list[Article]:
        articles: list[Article] = []
        for source in sources:
            try:
                raw = self._fetch(source.url)
                if source.kind == "rss":
                    articles.extend(self._parse_rss(source, raw, max_articles_per_source))
                elif source.kind == "html":
                    article = self._parse_article_page(source, source.url, raw, fallback_title=source.name)
                    if article:
                        articles.append(article)
                else:
                    LOGGER.warning("Skipping unsupported source kind %s for %s", source.kind, source.name)
            except Exception:
                LOGGER.exception("Failed to scrape source %s", source.url)
        return articles

    def _fetch(self, url: str) -> str:
        self.rate_limiter.wait()
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "GraphRAGFinancialNewsBot/0.1 (+https://example.local/bot)",
                "Accept": "text/html,application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not fetch {url}: {exc}") from exc

    def _parse_rss(
        self, source: NewsSource, xml_text: str, max_articles: int
    ) -> list[Article]:
        root = ET.fromstring(xml_text)
        items = root.findall(".//item")[:max_articles]
        articles: list[Article] = []

        for item in items:
            title = self.cleaner.clean_text(self._item_text(item, "title"))
            link = self._item_text(item, "link").strip()
            description = self.cleaner.clean_text(self._item_text(item, "description"))
            published_at = self._parse_datetime(
                self._item_text(item, "pubDate") or self._item_text(item, "published")
            )

            body = description
            if link:
                try:
                    article_html = self._fetch(link)
                    parsed = self._parse_article_page(source, link, article_html, fallback_title=title)
                    if parsed and parsed.body:
                        articles.append(
                            Article(
                                source_url=link,
                                title=parsed.title or title,
                                body=parsed.body,
                                published_at=parsed.published_at or published_at,
                                source=source.name,
                            )
                        )
                        continue
                except Exception:
                    LOGGER.info("Falling back to RSS description for %s", link, exc_info=True)

            if title and body:
                articles.append(
                    Article(
                        source_url=link or f"{source.url}#{len(articles)}",
                        title=title,
                        body=body,
                        published_at=published_at,
                        source=source.name,
                    )
                )
        return articles

    def _parse_article_page(
        self,
        source: NewsSource,
        url: str,
        html_text: str,
        fallback_title: str = "",
    ) -> Article | None:
        if BeautifulSoup is None:
            text = self.cleaner.clean_text(html_text)
            return Article(url, fallback_title, text, None, source.name) if text else None

        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer"]):
            tag.decompose()

        title_tag = soup.find("h1") or soup.find("title")
        title = self.cleaner.clean_text(title_tag.get_text(" ")) if title_tag else fallback_title

        if source.article_selector:
            containers = soup.select(source.article_selector)
        else:
            containers = soup.select("article") or [soup.body or soup]

        paragraphs: list[str] = []
        for container in containers:
            for paragraph in container.find_all(["p", "li"]):
                text = self.cleaner.clean_text(paragraph.get_text(" "))
                if len(text.split()) >= 5:
                    paragraphs.append(text)

        body = self.cleaner.clean_text(" ".join(paragraphs))
        if not body:
            return None

        published_at = None
        time_tag = soup.find("time")
        if time_tag:
            published_at = self._parse_datetime(
                time_tag.get("datetime") or time_tag.get_text(" ")
            )

        return Article(
            source_url=url,
            title=title,
            body=body,
            published_at=published_at,
            source=source.name,
        )

    @staticmethod
    def _item_text(item: ET.Element, tag_name: str) -> str:
        node = item.find(tag_name)
        if node is not None and node.text:
            return node.text
        return ""

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        value = (value or "").strip()
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            pass
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None
