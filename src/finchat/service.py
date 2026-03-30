from __future__ import annotations

from pathlib import Path
from typing import Optional

from .answering import answer_question
from .constants import NEWS_DATA_PATH
from .data import load_articles
from .llm import LLMClient, build_default_llm_client
from .models import AnswerResult
from .retrieval import RetrievalIndex, build_index, retrieve


class FinChatService:
    def __init__(self, index: RetrievalIndex, llm_client: Optional[LLMClient] = None) -> None:
        self.index = index
        self.llm_client = llm_client

    @classmethod
    def from_path(
        cls,
        data_path: Path = NEWS_DATA_PATH,
        llm_client: Optional[LLMClient] = None,
    ) -> "FinChatService":
        articles = load_articles(Path(data_path))
        index = build_index(articles)
        client = llm_client if llm_client is not None else build_default_llm_client()
        return cls(index=index, llm_client=client)

    @property
    def available_tickers(self) -> tuple[str, ...]:
        return tuple(sorted({article.ticker for article in self.index.articles}))

    @property
    def default_mode(self) -> str:
        return "llm" if self.llm_client is not None else "local"

    @property
    def retrieval_mode(self) -> str:
        return "Hash-vector retrieval"

    def answer(self, query: str, *, llm_client: Optional[LLMClient] = None) -> AnswerResult:
        contexts = retrieve(query, index=self.index)
        active_client = llm_client if llm_client is not None else self.llm_client
        return answer_question(query, contexts, llm_client=active_client)
