from __future__ import annotations

from typing import Optional, Sequence

from .constants import INSUFFICIENT_SUPPORT_MESSAGE
from .llm import LLMClient
from .local_summary import build_local_summary, is_supported_query
from .models import AnswerResult, RetrievedContext
from .retrieval import detect_tickers


def answer_question(
    query: str,
    contexts: Sequence[RetrievedContext],
    *,
    llm_client: Optional[LLMClient] = None,
) -> AnswerResult:
    matched_tickers = detect_tickers(query)
    if not is_supported_query(query, contexts):
        return AnswerResult(
            answer_text=(
                f"{INSUFFICIENT_SUPPORT_MESSAGE} "
                "Try asking about Apple, Microsoft, Amazon, Netflix, NVIDIA, Intel, IBM, or a specific event from the retrieved news."
            ),
            sources=tuple(contexts[:3]),
            mode="local",
            matched_tickers=matched_tickers,
        )

    if llm_client is not None:
        try:
            answer_text = llm_client.generate_answer(query, contexts)
        except Exception:
            answer_text = ""
        if answer_text.strip():
            return AnswerResult(
                answer_text=answer_text.strip(),
                sources=tuple(contexts[:4]),
                mode="llm",
                matched_tickers=matched_tickers,
            )

    return AnswerResult(
        answer_text=build_local_summary(query, contexts, matched_tickers),
        sources=tuple(contexts[:4]),
        mode="local",
        matched_tickers=matched_tickers,
    )
