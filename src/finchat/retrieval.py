"""In-memory chunk retrieval using hashed TF-IDF vectors over article body text."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha1
from math import log
from typing import Dict, List, Optional, Sequence, Tuple
import re

import numpy as np

from .constants import NEWS_DATA_PATH, TICKER_ALIASES
from .data import chunk_articles, deduplicate_articles, load_articles
from .models import Article, DocumentChunk, RetrievedContext


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+")
RETRIEVAL_STOPWORDS = {
    "about",
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "better",
    "by",
    "compare",
    "compared",
    "comparison",
    "coverage",
    "dataset",
    "doing",
    "for",
    "from",
    "has",
    "happening",
    "in",
    "is",
    "it",
    "its",
    "latest",
    "new",
    "news",
    "of",
    "on",
    "or",
    "recent",
    "recently",
    "summarize",
    "that",
    "the",
    "their",
    "than",
    "tell",
    "to",
    "versus",
    "vs",
    "what",
    "was",
    "were",
    "with",
}

DEFAULT_CHUNK_SIZE_WORDS = 120
DEFAULT_CHUNK_OVERLAP_WORDS = 30
EMBEDDING_DIMENSIONS = 256
PROJECTIONS_PER_TERM = 2

LOW_SIGNAL_CHUNK_PATTERNS = (
    "complete list of",
    "if you are looking for",
    "our conviction lies",
    "our methodology",
    "read next",
    "should you invest",
    "stock advisor",
    "story continues",
    "top ai stocks",
    "view comments",
    "while we acknowledge",
)

LOW_SIGNAL_TITLE_PATTERNS = (
    "bargain investors",
    "bull and bear of the day",
    "hand over fist",
    "jim cramer",
    "most profitable tech stock",
    "sell-off",
    "should you buy",
    "spotlight",
    "stock to buy now",
    "soars",
    "where will",
)

EVENTFUL_TITLE_HINTS = (
    "ai",
    "announces",
    "launch",
    "modem",
    "partnership",
    "planning",
    "product",
    "reported",
    "results",
    "teases",
)


@dataclass(frozen=True)
class VectorStore:
    idf_lookup: Dict[str, float]
    chunk_embeddings: np.ndarray
    embedding_dimensions: int


@dataclass(frozen=True)
class RetrievalIndex:
    articles: Tuple[Article, ...]
    chunks: Tuple[DocumentChunk, ...]
    vector_store: VectorStore
    article_lookup: Dict[str, Article]

    def retrieve(self, query: str, top_k: int = 8) -> List[RetrievedContext]:
        return retrieve(query, top_k=top_k, index=self)


def build_index(articles: Sequence[Article]) -> RetrievalIndex:
    unique_articles = deduplicate_articles(articles)
    chunks = chunk_articles(
        unique_articles,
        chunk_size_words=DEFAULT_CHUNK_SIZE_WORDS,
        overlap_words=DEFAULT_CHUNK_OVERLAP_WORDS,
    )
    tokenized_chunks = [_embedding_terms(chunk.text) for chunk in chunks]
    vector_store = _build_vector_store(tokenized_chunks)

    return RetrievalIndex(
        articles=tuple(unique_articles),
        chunks=tuple(chunks),
        vector_store=vector_store,
        article_lookup={article.id: article for article in unique_articles},
    )


def retrieve(
    query: str,
    top_k: int = 8,
    *,
    index: Optional[RetrievalIndex] = None,
    max_articles: int = 4,
) -> List[RetrievedContext]:
    """Return article-level contexts synthesized from the best-matching body-text chunks."""
    active_index = index or build_index(load_articles(NEWS_DATA_PATH))
    if not active_index.chunks:
        return []

    expanded_query, matched_tickers = expand_query(query)
    query_embedding = _embed_query(expanded_query, active_index.vector_store)
    if query_embedding is None:
        return []

    scores = _cosine_similarity_scores(query_embedding, active_index.vector_store.chunk_embeddings)
    scores = _apply_ticker_boost(scores, matched_tickers, active_index.chunks)
    scores = _apply_chunk_quality_adjustments(scores, query, matched_tickers, active_index.chunks)

    if not np.any(scores > 0):
        return []

    ranked_chunk_indexes = np.argsort(scores)[::-1][: max(top_k * 4, 16)]
    grouped_scores = defaultdict(lambda: {"max": 0.0, "sum": 0.0, "excerpts": []})

    for chunk_index in ranked_chunk_indexes:
        score = float(scores[chunk_index])
        if score <= 0:
            continue

        chunk = active_index.chunks[int(chunk_index)]
        bucket = grouped_scores[chunk.article_id]
        bucket["max"] = max(bucket["max"], score)
        bucket["sum"] += score
        if chunk.text not in bucket["excerpts"] and len(bucket["excerpts"]) < 3:
            bucket["excerpts"].append(chunk.text)

    ranked_articles: List[Tuple[str, float]] = []
    for article_id, bucket in grouped_scores.items():
        article = active_index.article_lookup[article_id]
        supporting_count = max(len(bucket["excerpts"]) - 1, 0)
        aggregate_score = bucket["max"] + ((bucket["sum"] - bucket["max"]) * 0.2) + (supporting_count * 0.03)
        aggregate_score += _article_quality_adjustment(query, article.title)
        ranked_articles.append((article_id, aggregate_score))

    ranked_articles.sort(key=lambda item: item[1], reverse=True)

    contexts: List[RetrievedContext] = []
    for article_id, aggregate_score in ranked_articles[:max_articles]:
        excerpts = grouped_scores[article_id]["excerpts"]
        if not excerpts:
            continue

        contexts.append(
            RetrievedContext(
                article=active_index.article_lookup[article_id],
                score=aggregate_score,
                excerpt=excerpts[0],
                supporting_excerpts=tuple(excerpts[1:]),
            )
        )

    return contexts


def expand_query(query: str) -> Tuple[str, Tuple[str, ...]]:
    matched_tickers = detect_tickers(query)
    expansions: List[str] = []

    for ticker in matched_tickers:
        expansions.extend(TICKER_ALIASES[ticker])
        expansions.append(ticker)

    expanded_query = " ".join(part for part in (query, " ".join(expansions)) if part.strip())
    return expanded_query, matched_tickers


def detect_tickers(query: str) -> Tuple[str, ...]:
    lowered = query.lower()
    matched: List[str] = []

    for ticker, aliases in TICKER_ALIASES.items():
        vocabulary = (ticker.lower(),) + tuple(alias.lower() for alias in aliases)
        if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in vocabulary):
            matched.append(ticker)

    return tuple(sorted(set(matched)))


def _embedding_terms(text: str) -> List[str]:
    tokens = [token for token in TOKEN_RE.findall(text.lower()) if token not in RETRIEVAL_STOPWORDS]
    bigrams = [f"{tokens[index]} {tokens[index + 1]}" for index in range(len(tokens) - 1)]
    return tokens + bigrams


def _build_vector_store(tokenized_chunks: Sequence[Sequence[str]]) -> VectorStore:
    document_frequency: Counter[str] = Counter()
    for terms in tokenized_chunks:
        document_frequency.update(set(terms))

    total_documents = len(tokenized_chunks) or 1
    idf_lookup = {
        term: log((1 + total_documents) / (1 + doc_count)) + 1.0
        for term, doc_count in document_frequency.items()
    }

    chunk_embeddings = np.zeros((len(tokenized_chunks), EMBEDDING_DIMENSIONS), dtype=np.float32)
    for row_index, terms in enumerate(tokenized_chunks):
        chunk_embeddings[row_index] = _hash_embed_terms(terms, idf_lookup)

    chunk_embeddings = _normalize_rows(chunk_embeddings)
    return VectorStore(
        idf_lookup=idf_lookup,
        chunk_embeddings=chunk_embeddings,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
    )


def _embed_query(query: str, vector_store: VectorStore) -> Optional[np.ndarray]:
    if not vector_store.idf_lookup:
        return None

    query_terms = [term for term in _embedding_terms(query) if term in vector_store.idf_lookup]
    if not query_terms:
        return None

    query_vector = _hash_embed_terms(query_terms, vector_store.idf_lookup, dimensions=vector_store.embedding_dimensions)
    query_vector = _normalize_vector(query_vector)
    if not np.any(query_vector):
        return None

    return query_vector.astype(np.float32)


def _cosine_similarity_scores(query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
    if chunk_embeddings.size == 0:
        return np.zeros(0, dtype=np.float32)
    return chunk_embeddings @ query_embedding


def _apply_ticker_boost(
    scores: np.ndarray,
    matched_tickers: Sequence[str],
    chunks: Sequence[DocumentChunk],
) -> np.ndarray:
    if not matched_tickers or scores.size == 0:
        return scores

    boosted = scores.copy()
    for index, chunk in enumerate(chunks):
        overlap = set(chunk.related_tickers or (chunk.ticker,)) & set(matched_tickers)
        if overlap:
            boosted[index] += 0.05 + (0.03 * min(len(overlap), 2))

    return boosted


def _apply_chunk_quality_adjustments(
    scores: np.ndarray,
    query: str,
    matched_tickers: Sequence[str],
    chunks: Sequence[DocumentChunk],
) -> np.ndarray:
    if scores.size == 0:
        return scores

    adjusted = scores.copy()
    topical_tokens = {token for token in TOKEN_RE.findall(query.lower()) if token not in RETRIEVAL_STOPWORDS}

    for index, chunk in enumerate(chunks):
        lowered = chunk.text.lower()
        adjustment = 0.0

        if any(pattern in lowered for pattern in LOW_SIGNAL_CHUNK_PATTERNS):
            adjustment -= 0.12

        matched_overlap = tuple(
            ticker for ticker in matched_tickers if ticker in (chunk.related_tickers or (chunk.ticker,))
        )
        if matched_overlap:
            mention_count = max(_ticker_mention_count(lowered, ticker) for ticker in matched_overlap)
            if mention_count >= 3:
                adjustment += 0.08
            elif mention_count == 2:
                adjustment += 0.05
            elif mention_count == 1:
                adjustment += 0.02
            else:
                adjustment -= 0.06

            if lowered.count("nasdaq:") >= 3:
                adjustment -= 0.05

        topical_overlap = sum(1 for token in topical_tokens if re.search(rf"\b{re.escape(token)}\b", lowered))
        adjustment += min(topical_overlap, 3) * 0.02
        adjusted[index] += adjustment

    return adjusted


def _hash_embed_terms(
    terms: Sequence[str],
    idf_lookup: Dict[str, float],
    *,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> np.ndarray:
    counts = Counter(term for term in terms if term in idf_lookup)
    vector = np.zeros(dimensions, dtype=np.float32)
    if not counts:
        return vector

    for term, count in counts.items():
        weight = (1.0 + log(count)) * float(idf_lookup[term])
        digest = sha1(term.encode("utf-8")).digest()
        for index in range(PROJECTIONS_PER_TERM):
            start = index * 4
            bucket = int.from_bytes(digest[start : start + 4], "big") % dimensions
            sign = 1.0 if digest[8 + index] % 2 == 0 else -1.0
            vector[bucket] += weight * sign

    return vector


def _ticker_mention_count(text: str, ticker: str) -> int:
    vocabulary = (ticker.lower(),) + tuple(alias.lower() for alias in TICKER_ALIASES[ticker])
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", text)) for term in vocabulary)


def _article_quality_adjustment(query: str, title: str) -> float:
    lowered_title = title.lower()
    lowered_query = query.lower()
    adjustment = 0.0

    if any(pattern in lowered_title for pattern in LOW_SIGNAL_TITLE_PATTERNS):
        adjustment -= 0.18
    if "?" in title:
        adjustment -= 0.06
    if "," in title and "highlights" in lowered_title:
        adjustment -= 0.08
    if any(hint in lowered_title for hint in EVENTFUL_TITLE_HINTS):
        adjustment += 0.05
    if any(term in lowered_query for term in ("latest", "recent")) and any(
        hint in lowered_title for hint in ("launch", "new product", "reported", "teases")
    ):
        adjustment += 0.06

    return adjustment


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)
