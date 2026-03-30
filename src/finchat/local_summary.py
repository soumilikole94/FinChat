"""Deterministic fallback summarization over noisy retrieved finance article text."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple
import re

from .constants import COMPANY_NAMES, TICKER_ALIASES
from .models import RetrievedContext
from .retrieval import detect_tickers


MIN_LOCAL_SUMMARY_POINTS = 3
MAX_LOCAL_SUMMARY_POINTS = 4

STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "across",
    "are",
    "as",
    "at",
    "better",
    "compare",
    "compared",
    "comparison",
    "coverage",
    "dataset",
    "did",
    "does",
    "doing",
    "for",
    "from",
    "happening",
    "in",
    "is",
    "it",
    "latest",
    "news",
    "new",
    "of",
    "on",
    "or",
    "recent",
    "recently",
    "say",
    "said",
    "show",
    "summarize",
    "tell",
    "the",
    "than",
    "these",
    "theme",
    "themes",
    "this",
    "to",
    "up",
    "versus",
    "vs",
    "what",
    "with",
}

ABSTRACT_TOPICAL_TOKENS = {
    "business",
    "challenge",
    "challenges",
    "company",
    "companies",
    "comparison",
    "coverage",
    "future",
    "happened",
    "happening",
    "hurdle",
    "main",
    "outlook",
    "performance",
    "plan",
    "plans",
    "position",
    "related",
    "risk",
    "risks",
    "show",
    "shows",
    "strategy",
    "strategies",
    "takeaway",
    "takeaways",
    "theme",
    "themes",
    "using",
}

BOILERPLATE_PATTERNS = (
    "hedge fund",
    "our methodology",
    "number of hedge fund",
    "read also",
    "read next",
    "story continues",
    "view comments",
)

LOW_SIGNAL_PATTERNS = (
    "10 ai stocks analysts are watching",
    "14 ai stocks on wall street",
    "already have a subscription",
    "all the rage",
    "cheapest ai stock",
    "contact:",
    "complete list of",
    "download multimedia",
    "free stock analysis report",
    "for immediate release",
    "getty images",
    "gurufocus",
    "high-flying ai stocks",
    "he opened with the following",
    "here is a synopsis of all five stocks",
    "in this article, we are going to take a look",
    "stands against the other stocks",
    "motley fool",
    "most profitable tech stock",
    "newsletter's strategy",
    "newsletter’s strategy",
    "originally appeared on",
    "photo:",
    "premium upgrade",
    "prnewswire",
    "source ibm",
    "stock advisor",
    "stock to buy now",
    "should you buy",
    "subscription plan",
    "times earnings",
    "view original content",
    "warning!",
    "hand over fist",
    "where will",
    "here is why",
    "if you are looking for",
    "investor letter",
    "while we acknowledge",
    "overall,",
    "number of hedge fund holders",
    "our list of",
    "retains buy rating",
    "a close-up of",
    "wide view of an apple store",
    "a closeup of",
    "bull of the day",
    "bear of the day",
)

LOW_SIGNAL_TITLE_TOKENS = {
    "coverage",
    "latest",
    "most",
    "news",
    "profitable",
    "returned",
    "stock",
    "stocks",
    "tech",
    "why",
    "year",
}

EVENT_HINTS = (
    "ai",
    "announced",
    "announcement",
    "assistant",
    "capacity",
    "chip",
    "collaboration",
    "deal",
    "demand",
    "growth",
    "jumped",
    "launch",
    "modem",
    "planning",
    "reported",
    "revenue",
    "split",
    "surged",
    "talks",
)

SPECULATIVE_PREFIXES = (
    "if ",
    "in particular, if",
    "there's no way to know",
    "unless you think",
    "while we acknowledge",
)

INVESTMENT_QUERY_TERMS = {
    "buy",
    "earnings",
    "forecast",
    "invest",
    "investing",
    "portfolio",
    "price",
    "rating",
    "shares",
    "stock",
    "target",
    "valuation",
}

INVESTMENT_NOISE_PATTERNS = (
    "equity portfolio",
    "hedge fund",
    "price target",
    "shares have",
    "shares were",
    "stock closed",
    "stock has",
    "target price",
)

COMPARISON_QUERY_PATTERNS = (
    "better than",
    "compare",
    "compared with",
    "compared to",
    "doing better",
    "versus",
    " vs ",
)

COMPARISON_SIGNAL_TERMS = (
    "alliance",
    "annualized revenue",
    "beat earnings expectations",
    "chip",
    "china",
    "demand",
    "earnings",
    "growth",
    "guidance",
    "intelligence",
    "launch",
    "market share",
    "modem",
    "net income",
    "outperform",
    "partnership",
    "product",
    "profit",
    "reported",
    "revenue",
    "sales",
    "surged",
)


def build_local_summary(
    query: str,
    contexts: Sequence[RetrievedContext],
    matched_tickers: Sequence[str],
) -> str:
    if is_comparison_query(query, matched_tickers):
        ordered_tickers = tickers_in_query_order(query, matched_tickers)
        return build_comparison_summary(query, contexts, ordered_tickers)

    return "\n".join(f"- {point}" for point in summary_points(query, contexts))


def is_supported_query(query: str, contexts: Sequence[RetrievedContext]) -> bool:
    if not contexts:
        return False

    top_score = contexts[0].score
    matched_tickers = detect_tickers(query)
    if is_comparison_query(query, matched_tickers):
        context_tickers = {
            ticker
            for context in contexts[:4]
            for ticker in (context.article.related_tickers or (context.article.ticker,))
        }
        if len(context_tickers & set(matched_tickers)) < 2:
            return False

    query_tokens = set(content_tokens(query))
    topical_tokens = topical_query_tokens(query)
    context_tokens = set()
    for context in contexts[:3]:
        context_tokens.update(content_tokens(context.article.title))
        context_tokens.update(content_tokens(context.excerpt))

    overlap = len(query_tokens & context_tokens)
    required_overlap = 1 if len(query_tokens) <= 3 else 2
    topical_overlap = len(topical_tokens & context_tokens)
    unmatched_topical_tokens = topical_tokens - context_tokens
    meaningful_unmatched_topical_tokens = {
        token for token in unmatched_topical_tokens if token not in ABSTRACT_TOPICAL_TOKENS
    }

    if has_unsupported_named_entity(query, contexts):
        return False

    if matched_tickers and meaningful_unmatched_topical_tokens:
        return False
    if not matched_tickers and topical_tokens and topical_overlap == 0:
        return False
    if not matched_tickers and len(topical_tokens) >= 2 and topical_overlap < 2:
        return False

    if top_score >= 0.28:
        return True

    if top_score >= 0.18 and overlap >= required_overlap:
        return True

    return False


def build_comparison_summary(
    query: str,
    contexts: Sequence[RetrievedContext],
    matched_tickers: Sequence[str],
) -> str:
    comparison_rows = comparison_points(query, contexts, matched_tickers)
    if len(comparison_rows) < 2:
        return "\n".join(f"- {point}" for point in summary_points(query, contexts))

    company_labels = [COMPANY_NAMES.get(ticker, ticker) for ticker in matched_tickers[:2]]
    intro = (
        f"The dataset does not provide a clean apples-to-apples benchmark for whether "
        f"{company_labels[0]} is doing better than {company_labels[1]} overall, but the retrieved coverage suggests:"
    )
    lines = [intro, ""]
    lines.extend(f"- {point}" for point in comparison_rows)
    return "\n".join(lines)


def comparison_points(
    query: str,
    contexts: Sequence[RetrievedContext],
    matched_tickers: Sequence[str],
) -> List[str]:
    points: List[str] = []
    point_limit = max(1, MAX_LOCAL_SUMMARY_POINTS // max(len(matched_tickers), 1))

    for ticker in matched_tickers:
        best_rows: List[Tuple[float, str, str]] = []
        seen_sentence_keys = set()
        ticker_contexts = [
            context
            for context in contexts[:4]
            if ticker in (context.article.related_tickers or (context.article.ticker,))
        ]
        for context in ticker_contexts:
            for sentence in comparison_candidate_sentences(query, context, ticker, matched_tickers):
                score = comparison_sentence_score(query, sentence, context.score, ticker)
                sentence_key = sentence_key_for(sentence)
                if sentence_key in seen_sentence_keys:
                    continue
                seen_sentence_keys.add(sentence_key)
                best_rows.append((score, context.article.id, sentence))

        if not best_rows:
            fallback_context = next(
                (
                    context
                    for context in contexts[:4]
                    if ticker in (context.article.related_tickers or (context.article.ticker,))
                ),
                None,
            )
            if fallback_context is not None:
                points.append(fallback_summary_point(fallback_context))
            continue

        company_label = COMPANY_NAMES.get(ticker, ticker)
        best_rows.sort(key=lambda item: item[0], reverse=True)
        lead_score = best_rows[0][0]
        used_articles = set()
        added_points = 0
        for score, article_id, sentence in best_rows:
            if article_id in used_articles:
                continue
            if added_points >= 1 and not should_include_optional_point(score, lead_score):
                break
            used_articles.add(article_id)
            points.append(f"{company_label} ({ticker}): {sentence}")
            added_points += 1
            if added_points == point_limit:
                break

        if len(points) >= MAX_LOCAL_SUMMARY_POINTS:
            break

    return points[:MAX_LOCAL_SUMMARY_POINTS]


def summary_points(query: str, contexts: Sequence[RetrievedContext]) -> List[str]:
    candidate_rows: List[Tuple[float, str, str, str]] = []
    unique_tickers = {context.article.ticker for context in contexts[:4]}
    per_ticker_limit = (
        MAX_LOCAL_SUMMARY_POINTS
        if len(unique_tickers) == 1
        else max(1, MAX_LOCAL_SUMMARY_POINTS // max(len(unique_tickers), 1))
    )
    per_article_limit = 2 if len(unique_tickers) == 1 else 1

    for context in contexts[:4]:
        for sentence, anchored, theme_overlap in candidate_sentences(query, context):
            score = score_sentence(
                query,
                sentence,
                context.score,
                anchored=anchored,
                theme_overlap=theme_overlap,
            )
            if score <= 0:
                continue
            candidate_rows.append((score, context.article.ticker, context.article.id, sentence))

    if not candidate_rows:
        return [fallback_summary_point(context) for context in contexts[:MAX_LOCAL_SUMMARY_POINTS]]

    candidate_rows.sort(key=lambda item: item[0], reverse=True)
    lead_score = candidate_rows[0][0]

    seen_sentence_keys = set()
    ticker_counts = defaultdict(int)
    article_counts = defaultdict(int)
    points: List[str] = []

    for score, ticker, article_id, sentence in candidate_rows:
        if ticker_counts[ticker] >= per_ticker_limit:
            continue
        if article_counts[article_id] >= per_article_limit:
            continue
        if len(points) >= MIN_LOCAL_SUMMARY_POINTS and not should_include_optional_point(score, lead_score):
            break

        sentence_key = sentence_key_for(sentence)
        if sentence_key in seen_sentence_keys:
            continue

        seen_sentence_keys.add(sentence_key)
        ticker_counts[ticker] += 1
        article_counts[article_id] += 1
        company_label = COMPANY_NAMES.get(ticker, ticker)
        points.append(f"{company_label} ({ticker}): {sentence}")

        if len(points) == MAX_LOCAL_SUMMARY_POINTS:
            break

    return points or [fallback_summary_point(context) for context in contexts[:MAX_LOCAL_SUMMARY_POINTS]]


def candidate_sentences(query: str, context: RetrievedContext) -> List[Tuple[str, bool, int]]:
    candidates: List[Tuple[str, bool, int]] = []
    seen = set()
    title_theme_tokens = article_theme_tokens(context.article.title, context.article.ticker)

    blocks = [context.excerpt, *context.supporting_excerpts]
    for block in blocks:
        for sentence in extract_sentences(block):
            sentence_tokens = set(content_tokens(sentence))
            theme_overlap = len(sentence_tokens & title_theme_tokens)
            if not is_relevant_article_sentence(query, sentence, context.article.ticker, title_theme_tokens, anchored=True):
                continue
            sentence_key = sentence_key_for(sentence)
            if sentence_key in seen:
                continue
            seen.add(sentence_key)
            candidates.append((sentence, True, theme_overlap))

    return candidates


def comparison_candidate_sentences(
    query: str,
    context: RetrievedContext,
    ticker: str,
    matched_tickers: Sequence[str],
) -> List[str]:
    candidates: List[str] = []
    seen = set()
    title_theme_tokens = article_theme_tokens(context.article.title, ticker)
    title_mentions_ticker = company_mention_count(context.article.title, ticker) > 0

    blocks = [context.excerpt, *context.supporting_excerpts]
    for block in blocks:
        for sentence in extract_sentences(block):
            sentence_key = sentence_key_for(sentence)
            if sentence_key in seen:
                continue
            sentence_tokens = set(content_tokens(sentence))
            theme_overlap = len(sentence_tokens & title_theme_tokens)
            mention_count = company_mention_count(sentence, ticker)
            if mention_count == 0:
                continue
            if has_competing_lead_company(sentence, ticker, matched_tickers):
                continue
            if (
                not title_mentions_ticker
                and any(company_mention_count(context.article.title, other_ticker) > 0 for other_ticker in matched_tickers if other_ticker != ticker)
            ):
                continue
            if (
                not is_investment_focused_query(query)
                and any(pattern in sentence.lower() for pattern in INVESTMENT_NOISE_PATTERNS)
            ):
                continue
            if theme_overlap == 0 and not any(term in sentence.lower() for term in COMPARISON_SIGNAL_TERMS):
                continue
            seen.add(sentence_key)
            candidates.append(sentence)

    return candidates


def fallback_summary_point(context: RetrievedContext) -> str:
    company_label = COMPANY_NAMES.get(context.article.ticker, context.article.ticker)
    return f"{company_label} ({context.article.ticker}): {ensure_sentence_ending(context.article.title)}"


def score_sentence(
    query: str,
    sentence: str,
    retrieval_score: float,
    *,
    anchored: bool = False,
    theme_overlap: int = 0,
) -> float:
    query_tokens = set(content_tokens(query))
    topical_tokens = topical_query_tokens(query)
    sentence_tokens = set(content_tokens(sentence))
    overlap = len(query_tokens & sentence_tokens)
    topical_overlap = len(topical_tokens & sentence_tokens)
    anchor_bonus = 0.35 if anchored else 0.0
    theme_bonus = theme_overlap * 0.18
    event_bonus = 0.18 if any(term in sentence.lower() for term in EVENT_HINTS) else 0.0
    freshness_bonus = 0.16 if looks_recent(query, sentence) else 0.0
    detail_bonus = 0.12 if re.search(r"\d", sentence) else 0.0
    length_bonus = 0.08 if 8 <= len(sentence.split()) <= 35 else 0.0
    investment_penalty = (
        0.32
        if (not is_investment_focused_query(query) and any(pattern in sentence.lower() for pattern in INVESTMENT_NOISE_PATTERNS))
        else 0.0
    )
    quote_penalty = 0.25 if any(mark in sentence for mark in ('"', "“", "”", "‘", "’")) else 0.0
    title_penalty = 0.45 if looks_clickbaity(sentence) else 0.0
    topical_miss_penalty = 0.22 if topical_tokens and topical_overlap == 0 else 0.0
    return (
        retrieval_score
        + anchor_bonus
        + theme_bonus
        + (overlap * 0.18)
        + (topical_overlap * 0.32)
        + event_bonus
        + freshness_bonus
        + detail_bonus
        + length_bonus
        - investment_penalty
        - quote_penalty
        - title_penalty
        - topical_miss_penalty
    )


def comparison_sentence_score(query: str, sentence: str, retrieval_score: float, ticker: str) -> float:
    sentence_tokens = set(content_tokens(sentence))
    ticker_terms = {ticker.lower(), *(alias.lower() for alias in TICKER_ALIASES[ticker])}
    ticker_bonus = 0.18 if sentence_tokens & ticker_terms else 0.0
    signal_bonus = 0.3 if any(term in sentence.lower() for term in COMPARISON_SIGNAL_TERMS) else 0.0
    detail_bonus = 0.12 if re.search(r"\d", sentence) else 0.0
    lead_bonus = 0.16 if company_mentioned_first(sentence, ticker) else 0.0
    investment_penalty = (
        0.28
        if (not is_investment_focused_query(query) and any(pattern in sentence.lower() for pattern in INVESTMENT_NOISE_PATTERNS))
        else 0.0
    )
    title_penalty = 0.4 if looks_clickbaity(sentence) else 0.0
    return (retrieval_score * 0.55) + ticker_bonus + signal_bonus + detail_bonus + lead_bonus - investment_penalty - title_penalty


def should_include_optional_point(score: float, lead_score: float) -> bool:
    return score >= max(0.6, lead_score * 0.72)


def company_mention_count(text: str, ticker: str) -> int:
    vocabulary = (ticker.lower(),) + tuple(alias.lower() for alias in TICKER_ALIASES[ticker])
    lowered = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", lowered)) for term in vocabulary)


def has_competing_lead_company(sentence: str, ticker: str, matched_tickers: Sequence[str]) -> bool:
    lowered = sentence.lower()
    target_positions = company_mention_positions(lowered, ticker)
    if not target_positions:
        return True

    target_first = min(target_positions)
    for other_ticker in matched_tickers:
        if other_ticker == ticker:
            continue
        other_positions = company_mention_positions(lowered, other_ticker)
        if other_positions and min(other_positions) < target_first:
            return True

    return False


def company_mentioned_first(sentence: str, ticker: str) -> bool:
    positions = company_mention_positions(sentence.lower(), ticker)
    if not positions:
        return False
    return min(positions) <= 20


def company_mention_positions(lowered_text: str, ticker: str) -> List[int]:
    positions: List[int] = []
    for term in (ticker.lower(),) + tuple(alias.lower() for alias in TICKER_ALIASES[ticker]):
        positions.extend(match.start() for match in re.finditer(rf"\b{re.escape(term)}\b", lowered_text))
    return positions


def split_sentences(text: str) -> List[str]:
    protected = text or ""
    protected = re.sub(r"\b(Inc|Corp|Co|Ltd|Mr|Mrs|Ms|Dr|Prof)\.", r"\1<prd>", protected)
    protected = protected.replace("U.S.", "U<prd>S<prd>")
    parts = re.split(r"(?<=[.!?])\s+", protected)
    parts = [part.replace("<prd>", ".") for part in parts]
    return [part.strip() for part in parts if part.strip()]


def extract_sentences(text: str) -> List[str]:
    extracted: List[str] = []
    for sentence in split_sentences(text):
        if not sentence.endswith((".", "!", "?")):
            continue
        cleaned = normalize_summary_sentence(clean_sentence(sentence))
        if is_informative_sentence(cleaned):
            extracted.append(ensure_sentence_ending(cleaned))
    return extracted


def clean_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().strip("-")


def normalize_summary_sentence(sentence: str) -> str:
    normalized = sentence
    normalized = re.sub(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+Inc\.\s*[’']s\b", r"\1's", normalized)
    normalized = re.sub(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+Corp\.\s*[’']s\b", r"\1's", normalized)
    normalized = re.sub(
        r"\((?:NASDAQ|NYSE|NYSEARCA|NYSEAMERICAN|AMEX|OTC|NASDAQGS|NASDAQGM):[A-Z0-9._-]+\)",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+[’']s\b", "'s", normalized)
    normalized = re.sub(
        r"^[A-Z][A-Za-z .'-]+,\s*[A-Z. ]+,\s*[A-Za-z]{3,9}\.\s+\d{1,2},\s+\d{4}\s*/PRNewswire/\s*--\s*",
        "",
        normalized,
    )
    normalized = re.sub(r"^[A-Za-z]{3,9}\.\s+\d{1,2},\s+\d{4}\s+", "", normalized)
    normalized = re.sub(r"\(([A-Z]{1,5},\s*Financials)\)", "", normalized)
    normalized = re.sub(
        r"^[A-Z][A-Za-z ]+:\s+The stock [^.?!]*?\b([A-Z][A-Za-z]+)\b\s+(specializes in)",
        r"\1 \2",
        normalized,
    )
    normalized = re.sub(
        r"^[A-Z][A-Za-z '&.-]+ stated the following regarding [^.]+?:\s*[\"“”‘’]*",
        "",
        normalized,
    )
    normalized = re.sub(r"^[A-Z][a-z]{2,8}\.\s+\d{1,2},\s+\d{4}\s+[A-Z][a-z]+ reported that\s+", "", normalized)
    normalized = re.sub(r"^[A-Z][a-z]+ News further reported that\s+", "", normalized)
    normalized = re.sub(r"^[A-Z][a-z]+ reported that\s+", "", normalized)
    normalized = re.sub(
        r"^(?:Moreover,\s+)?the model will also be ([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)(?: Inc\.)?'s first device with ",
        r"\1's new device will also be its first device with ",
        normalized,
    )
    normalized = re.sub(r"^[\"']+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" -\"“”‘’")


def ensure_sentence_ending(sentence: str) -> str:
    if sentence.endswith((".", "!", "?")):
        return sentence
    return sentence + "."


def is_boilerplate_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(pattern in lowered for pattern in BOILERPLATE_PATTERNS)


def is_informative_sentence(sentence: str, allow_title: bool = False) -> bool:
    if not sentence:
        return False

    lowered = sentence.lower()
    words = sentence.split()
    if len(words) < 7:
        return False
    if not sentence[0].isalpha() or sentence[0].islower():
        return False
    if lowered.endswith(("through.", "with.", "for.", "of.", "to.", "and.", "or.")):
        return False
    if re.search(r"https?://|www\.|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", sentence):
        return False
    if sum(sentence.count(mark) for mark in ('"', "“", "”", "‘", "’")) >= 2:
        return False
    if lowered.startswith(
        ("i ", "i'm", "i am", "we ", "well ", "and i think", "but we don't know", "here’s what to know", "here's what to know")
    ):
        return False
    if is_boilerplate_sentence(sentence) or looks_clickbaity(sentence):
        return False
    if any(lowered.startswith(prefix) for prefix in SPECULATIVE_PREFIXES):
        return False
    if "?" in sentence and not allow_title:
        return False
    if any(lowered.startswith(prefix) for prefix in ("overall,", "while we acknowledge", "if you are looking")):
        return False
    if "/prnewswire/" in lowered:
        return False

    return True


def looks_clickbaity(sentence: str) -> bool:
    lowered = sentence.lower()
    if any(pattern in lowered for pattern in LOW_SIGNAL_PATTERNS):
        return True
    if lowered.startswith(("is ", "should you", "where will")):
        return True
    return "?" in sentence


def sentence_key_for(sentence: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()


def is_relevant_article_sentence(
    query: str,
    sentence: str,
    ticker: str,
    title_theme_tokens: set,
    *,
    anchored: bool,
) -> bool:
    topical_tokens = topical_query_tokens(query)
    matched_tickers = detect_tickers(query)
    sentence_tokens = set(content_tokens(sentence))
    alias_tokens = {ticker.lower(), *(alias.lower() for alias in TICKER_ALIASES[ticker])}
    theme_overlap = len(sentence_tokens & title_theme_tokens)
    has_alias_mention = bool(sentence_tokens & alias_tokens)
    has_signal_detail = bool(re.search(r"\d", sentence)) or any(term in sentence.lower() for term in EVENT_HINTS)

    if topical_tokens:
        if sentence_tokens & topical_tokens:
            return True
        return anchored and theme_overlap >= 2

    if matched_tickers:
        if has_alias_mention and (theme_overlap >= 1 or has_signal_detail):
            return True
        return anchored and theme_overlap >= 2 and has_signal_detail

    if theme_overlap >= 2:
        return True

    return anchored and has_alias_mention and has_signal_detail


def article_theme_tokens(title: str, ticker: str) -> set:
    tokens = set(content_tokens(title))
    tokens -= LOW_SIGNAL_TITLE_TOKENS
    tokens.discard(ticker.lower())
    for alias in TICKER_ALIASES[ticker]:
        tokens.discard(alias.lower())
    return tokens


def topical_query_tokens(query: str) -> set:
    tokens = set(content_tokens(query))
    for ticker in detect_tickers(query):
        tokens.discard(ticker.lower())
        for alias in TICKER_ALIASES[ticker]:
            tokens.discard(alias.lower())
    return tokens


def is_comparison_query(query: str, matched_tickers: Sequence[str]) -> bool:
    if len(matched_tickers) < 2:
        return False
    normalized_query = f" {query.lower()} "
    return any(pattern in normalized_query for pattern in COMPARISON_QUERY_PATTERNS)


def tickers_in_query_order(query: str, matched_tickers: Sequence[str]) -> Tuple[str, ...]:
    lowered_query = query.lower()
    ranked_tickers = []

    for ticker in matched_tickers:
        aliases = (ticker.lower(),) + tuple(alias.lower() for alias in TICKER_ALIASES[ticker])
        positions = [lowered_query.find(alias) for alias in aliases if lowered_query.find(alias) != -1]
        ranked_tickers.append((min(positions) if positions else len(lowered_query), ticker))

    return tuple(ticker for _, ticker in sorted(ranked_tickers))


def has_unsupported_named_entity(query: str, contexts: Sequence[RetrievedContext]) -> bool:
    capitalized_words = re.findall(r"\b[A-Z][A-Za-z0-9&.'-]+\b", query)
    known_company_tokens = set()
    for ticker, aliases in TICKER_ALIASES.items():
        known_company_tokens.add(ticker.lower())
        known_company_tokens.update(alias.lower() for alias in aliases)

    title_text = " ".join(context.article.title for context in contexts[:4]).lower()
    for word in capitalized_words:
        lowered = word.lower()
        if lowered.endswith("'s") and len(lowered) > 2:
            lowered = lowered[:-2]
        if lowered in STOPWORDS or lowered in known_company_tokens:
            continue
        if lowered not in title_text:
            return True

    return False


def content_tokens(text: str) -> List[str]:
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9']+", text.lower())
    normalized_tokens = []
    for token in raw_tokens:
        if token.endswith("'s") and len(token) > 2:
            token = token[:-2]
        normalized_tokens.append(token)
    return [token for token in normalized_tokens if token and token not in STOPWORDS]


def is_investment_focused_query(query: str) -> bool:
    return any(term in set(content_tokens(query)) for term in INVESTMENT_QUERY_TERMS)


def looks_recent(query: str, sentence: str) -> bool:
    if not any(term in query.lower() for term in ("latest", "recent")):
        return False
    return bool(
        re.search(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}\b",
            sentence.lower(),
        )
    )
