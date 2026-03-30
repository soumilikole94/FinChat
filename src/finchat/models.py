from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Article:
    id: str
    ticker: str
    title: str
    link: str
    full_text: str
    related_tickers: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    article_id: str
    ticker: str
    article_title: str
    text: str
    ordinal: int
    related_tickers: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedContext:
    article: Article
    score: float
    excerpt: str
    supporting_excerpts: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerResult:
    answer_text: str
    sources: Tuple[RetrievedContext, ...]
    mode: str
    matched_tickers: Tuple[str, ...]
