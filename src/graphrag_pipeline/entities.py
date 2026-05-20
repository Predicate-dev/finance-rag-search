from __future__ import annotations

import math
import re
from collections import defaultdict

from graphrag_pipeline.domain import Article, Entity, Relation, SentimentBreakdown


class FinancialSentimentAnalyzer:
    """Rule-based, finance-aware sentiment scorer in [-1, 1].

    This is intentionally transparent and trainable: production systems can replace the
    lexicon or add supervised calibration without changing graph construction.
    """

    POSITIVE = {
        "beat",
        "beats",
        "bullish",
        "upgrade",
        "upgraded",
        "growth",
        "profit",
        "profitable",
        "surge",
        "rally",
        "record",
        "strong",
        "resilient",
        "eased",
        "raises",
        "outperform",
        "buyback",
    }
    NEGATIVE = {
        "miss",
        "misses",
        "bearish",
        "downgrade",
        "downgraded",
        "loss",
        "decline",
        "plunge",
        "weak",
        "lawsuit",
        "probe",
        "cuts",
        "layoff",
        "risk",
        "pressure",
        "shortage",
        "underperform",
    }
    NEGATORS = {"not", "never", "no", "without"}

    TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+|[$]?[A-Z]{1,5}\b|[-+]?\d+(?:\.\d+)?%?")

    def score(self, text: str) -> float:
        tokens = [token.lower().strip("$") for token in self.TOKEN_RE.findall(text)]
        if not tokens:
            return 0.0
        raw = 0.0
        evidence_terms = 0
        for index, token in enumerate(tokens):
            sign = 0
            if token in self.POSITIVE:
                sign = 1
            elif token in self.NEGATIVE:
                sign = -1
            if sign == 0:
                continue
            window = tokens[max(0, index - 3) : index]
            if any(item in self.NEGATORS for item in window):
                sign *= -1
            raw += sign
            evidence_terms += 1
        if evidence_terms == 0:
            return 0.0
        return max(-1.0, min(1.0, raw / math.sqrt(evidence_terms + 2)))


class FinancialEntityExtractor:
    """Extracts companies, tickers, people, events, and simple financial relations."""

    KNOWN_COMPANIES = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Nvidia": "NVDA",
        "Tesla": "TSLA",
        "Amazon": "AMZN",
        "Meta": "META",
        "Alphabet": "GOOGL",
        "Google": "GOOGL",
        "AMD": "AMD",
        "Intel": "INTC",
        "JPMorgan": "JPM",
        "Bank of America": "BAC",
    }
    TICKER_RE = re.compile(r"(?<![A-Z])(?:NYSE:|NASDAQ:|AMEX:|\$)?([A-Z]{1,5})(?![A-Z])")
    COMPANY_RE = re.compile(
        r"\b([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,4}\s+"
        r"(?:Inc|Corp|Corporation|Co|Company|Ltd|LLC|PLC|Group|Holdings))\b\.?"
    )
    PERSON_RE = re.compile(
        r"\b(?:CEO|CFO|COO|Chair|Analyst|President)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"
    )
    EVENT_KEYWORDS = {
        "earnings": "EARNINGS",
        "acquisition": "ACQUISITION",
        "merger": "MERGER",
        "supply chain": "SUPPLY_CHAIN",
        "lawsuit": "LEGAL",
        "guidance": "GUIDANCE",
        "buyback": "CAPITAL_RETURN",
        "dividend": "CAPITAL_RETURN",
    }
    COMMON_FALSE_TICKERS = {
        "CEO",
        "CFO",
        "COO",
        "EPS",
        "GDP",
        "SEC",
        "USA",
        "US",
        "AI",
        "EV",
        "IPO",
        "ETF",
        "Fed",
    }
    RELATION_PATTERNS = [
        (re.compile(r"\b(acquires?|acquired|buying|bought)\b", re.IGNORECASE), "ACQUIRES"),
        (re.compile(r"\b(partners?|partnered|collaborates?)\b", re.IGNORECASE), "PARTNERS_WITH"),
        (re.compile(r"\b(sues?|lawsuit|probe|investigation)\b", re.IGNORECASE), "LEGAL_ACTION"),
        (re.compile(r"\b(supplies|supplier|supply chain)\b", re.IGNORECASE), "SUPPLY_CHAIN_LINK"),
        (re.compile(r"\b(beats?|misses?)\b", re.IGNORECASE), "REPORTS_EARNINGS"),
    ]

    def __init__(self, sentiment: FinancialSentimentAnalyzer | None = None) -> None:
        self.sentiment = sentiment or FinancialSentimentAnalyzer()

    def extract_entities(self, text: str) -> list[Entity]:
        seen: dict[str, Entity] = {}
        for match in self.COMPANY_RE.finditer(text):
            name = match.group(1).strip()
            entity = self._entity(name=name, entity_type="COMPANY", text=text)
            seen[entity.canonical_id] = entity

        for name, ticker in self.KNOWN_COMPANIES.items():
            if self._mentions(text, name) or self._mentions(text, ticker):
                company = self._entity(name=name, entity_type="COMPANY", text=text)
                seen[company.canonical_id] = company
                ticker_entity = self._entity(name=ticker, entity_type="TICKER", text=text)
                seen[ticker_entity.canonical_id] = ticker_entity

        for match in self.TICKER_RE.finditer(text):
            ticker = match.group(1)
            if ticker in self.COMMON_FALSE_TICKERS or ticker.title() == ticker:
                continue
            entity = self._entity(name=ticker, entity_type="TICKER", text=text)
            seen[entity.canonical_id] = entity

        for match in self.PERSON_RE.finditer(text):
            name = match.group(1).strip()
            entity = self._entity(name=name, entity_type="PERSON", text=text)
            seen[entity.canonical_id] = entity

        lower_text = text.lower()
        for phrase, event_type in self.EVENT_KEYWORDS.items():
            if phrase in lower_text:
                name = event_type.replace("_", " ").title()
                entity = self._entity(name=name, entity_type="EVENT", text=text)
                seen[entity.canonical_id] = entity

        return sorted(seen.values(), key=lambda item: (item.entity_type, item.name))

    def extract_relations(self, text: str, article_id: str | None = None) -> list[Relation]:
        entities = self.extract_entities(text)
        if len(entities) < 2:
            return []

        sentences = split_sentences(text)
        relations: list[Relation] = []
        for sentence in sentences:
            sentence_entities = [
                entity for entity in entities if self._mentions(sentence, entity.name)
            ]
            if len(sentence_entities) < 2:
                continue
            for pattern, relation_type in self.RELATION_PATTERNS:
                if pattern.search(sentence):
                    source, target = sentence_entities[0], sentence_entities[1]
                    relations.append(
                        Relation(
                            source=source,
                            target=target,
                            relation_type=relation_type,
                            confidence=0.78,
                            sentiment=self.sentiment.score(sentence),
                            evidence=sentence,
                            article_id=article_id,
                        )
                    )
                    break
            else:
                source, target = sentence_entities[0], sentence_entities[1]
                relations.append(
                    Relation(
                        source=source,
                        target=target,
                        relation_type="MENTIONED_WITH",
                        confidence=0.45,
                        sentiment=self.sentiment.score(sentence),
                        evidence=sentence,
                        article_id=article_id,
                    )
                )
        return relations

    def sentiment_breakdown(self, text: str, relations: list[Relation]) -> SentimentBreakdown:
        entities = self.extract_entities(text)
        by_entity = {
            entity.name: self.sentiment.score(" ".join(context_windows(text, entity.name)))
            for entity in entities
        }
        by_relation: dict[str, float] = {}
        for relation in relations:
            key = f"{relation.source.name} {relation.relation_type} {relation.target.name}"
            by_relation[key] = relation.sentiment
        return SentimentBreakdown(
            overall=self.sentiment.score(text),
            by_entity=by_entity,
            by_relation=by_relation,
        )

    def _entity(self, name: str, entity_type: str, text: str) -> Entity:
        canonical = f"{entity_type}:{normalize_key(name)}"
        local_context = " ".join(context_windows(text, name))
        return Entity(
            name=name,
            entity_type=entity_type,
            canonical_id=canonical,
            sentiment=self.sentiment.score(local_context or text),
            metadata={"extractor": "rule_based_v1"},
        )

    @staticmethod
    def _mentions(text: str, name: str) -> bool:
        return re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE) is not None


class KnowledgeGraphBuilder:
    """Transforms an article into graph nodes and relation edges."""

    def __init__(self, extractor: FinancialEntityExtractor) -> None:
        self.extractor = extractor

    def build_for_article(self, article: Article, article_id: str, repository: object) -> SentimentBreakdown:
        text = f"{article.title}. {article.body}"
        entities = self.extractor.extract_entities(text)
        relations = self.extractor.extract_relations(text, article_id=article_id)

        for entity in entities:
            repository.upsert_entity(entity)

        # Relation endpoints may have been re-extracted while scanning sentences; upsert them too.
        for relation in relations:
            repository.upsert_entity(relation.source)
            repository.upsert_entity(relation.target)
            repository.upsert_relation(relation)

        return self.extractor.sentiment_breakdown(text, relations)


def split_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]


def context_windows(text: str, needle: str, radius: int = 1) -> list[str]:
    sentences = split_sentences(text)
    windows: list[str] = []
    for index, sentence in enumerate(sentences):
        if re.search(rf"\b{re.escape(needle)}\b", sentence, re.IGNORECASE):
            start = max(0, index - radius)
            end = min(len(sentences), index + radius + 1)
            windows.append(" ".join(sentences[start:end]))
    return windows


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
