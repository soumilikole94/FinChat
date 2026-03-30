from __future__ import annotations

from hashlib import sha1
from html import unescape
from pathlib import Path
from typing import Dict, List, Sequence
import json
import re
import unicodedata

from .constants import TICKER_ALIASES
from .models import Article, DocumentChunk


TRUNCATE_MARKERS = (
    "READ NEXT:",
    "Disclosure:",
    "View Comments",
)

INLINE_MARKERS = (
    "Story Continues",
    "READ ALSO:",
)


def load_articles(path: Path) -> List[Article]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    articles: List[Article] = []

    for ticker, items in raw.items():
        for item in items:
            title = normalize_whitespace(item["title"])
            link = normalize_whitespace(item["link"])
            full_text = clean_article_text(item.get("full_text", ""))
            article_id = build_article_id(ticker, title, link)
            articles.append(
                Article(
                    id=article_id,
                    ticker=ticker,
                    title=title,
                    link=link,
                    full_text=full_text,
                    related_tickers=(ticker,),
                )
            )

    return articles


def clean_article_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", unescape(text or ""))
    cleaned = cleaned.replace("\xa0", " ")

    for marker in TRUNCATE_MARKERS:
        marker_index = cleaned.find(marker)
        if marker_index != -1:
            cleaned = cleaned[:marker_index]

    for marker in INLINE_MARKERS:
        cleaned = cleaned.replace(marker, " ")

    cleaned = re.sub(r"This article is originally published at[^.]*\.", " ", cleaned, flags=re.IGNORECASE)
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def deduplicate_articles(articles: Sequence[Article]) -> List[Article]:
    grouped: Dict[str, List[Article]] = {}

    for article in articles:
        grouped.setdefault(article_dedup_key(article), []).append(article)

    deduped: List[Article] = []
    for article_group in grouped.values():
        base_article = article_group[0]
        related_tickers = tuple(sorted({ticker for article in article_group for ticker in article.related_tickers or (article.ticker,)}))
        primary_ticker = infer_primary_ticker(base_article.title, base_article.full_text, related_tickers, fallback=base_article.ticker)
        deduped.append(
            Article(
                id=build_article_id(primary_ticker, base_article.title, base_article.link),
                ticker=primary_ticker,
                title=base_article.title,
                link=base_article.link,
                full_text=base_article.full_text,
                related_tickers=related_tickers,
            )
        )

    return deduped


def chunk_articles(
    articles: Sequence[Article],
    chunk_size_words: int = 180,
    overlap_words: int = 40,
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    step = max(chunk_size_words - overlap_words, 1)

    for article in articles:
        words = article.full_text.split()
        if not words:
            continue

        for ordinal, start in enumerate(range(0, len(words), step)):
            window = words[start : start + chunk_size_words]
            if not window:
                continue
            chunk_text = " ".join(window).strip()
            chunk_id = f"{article.id}-chunk-{ordinal}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    article_id=article.id,
                    ticker=article.ticker,
                    article_title=article.title,
                    text=chunk_text,
                    ordinal=ordinal,
                    related_tickers=article.related_tickers or (article.ticker,),
                )
            )

            if start + chunk_size_words >= len(words):
                break

    return chunks


def article_dedup_key(article: Article) -> str:
    title_key = normalize_for_key(article.title)
    link_key = normalize_for_key(article.link)
    return f"{title_key}|{link_key}"


def build_article_id(ticker: str, title: str, link: str) -> str:
    digest = sha1(f"{ticker}|{title}|{link}".encode("utf-8")).hexdigest()[:12]
    return f"{ticker.lower()}-{digest}"


def infer_primary_ticker(title: str, full_text: str, candidate_tickers: Sequence[str], *, fallback: str) -> str:
    title_lower = title.lower()
    body_window = full_text[:1500].lower()
    scored_candidates = []

    for ticker in candidate_tickers:
        aliases = (ticker.lower(),) + tuple(alias.lower() for alias in TICKER_ALIASES[ticker])
        title_mentions = sum(len(re.findall(rf"\b{re.escape(alias)}\b", title_lower)) for alias in aliases)
        body_mentions = sum(len(re.findall(rf"\b{re.escape(alias)}\b", body_window)) for alias in aliases)
        positions = [title_lower.find(alias) for alias in aliases if title_lower.find(alias) != -1]
        first_position = min(positions) if positions else len(title_lower) + 100
        score = (title_mentions * 8) + min(body_mentions, 4) - (first_position * 0.01)
        scored_candidates.append((score, ticker))

    scored_candidates.sort(reverse=True)
    best_score, best_ticker = scored_candidates[0] if scored_candidates else (0.0, fallback)
    if best_score <= 0:
        return fallback
    return best_ticker


def normalize_for_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    normalized = re.sub(r"\W+", " ", normalized)
    return normalized.strip()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()
