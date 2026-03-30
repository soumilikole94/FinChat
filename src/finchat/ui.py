from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .constants import DEFAULT_ANTHROPIC_MODEL, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL
from .llm import LLMClient, LLMConfig, build_llm_client
from .service import FinChatService


SAMPLE_QUESTIONS = (
    "What is the latest Apple news in the dataset?",
    "Summarize the NVIDIA coverage.",
    "What is happening with Intel?",
    "Compare NVIDIA and Apple based on the dataset.",
    "What does the dataset say about Apple's AI plans in China?",
    "How is IBM using AI in the dataset?",
    "Is Apple the company related to the fruit apple?",
)

SESSION_LLM_OPTIONS = {
    "Local only": {"provider": None, "default_model": "", "requires_base_url": False},
    "OpenAI": {"provider": "openai", "default_model": DEFAULT_OPENAI_MODEL, "requires_base_url": False},
    "Anthropic": {"provider": "anthropic", "default_model": DEFAULT_ANTHROPIC_MODEL, "requires_base_url": False},
    "Gemini": {"provider": "gemini", "default_model": DEFAULT_GEMINI_MODEL, "requires_base_url": False},
    "OpenAI-compatible": {"provider": "openai", "default_model": DEFAULT_OPENAI_MODEL, "requires_base_url": True},
}


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="FinChat", page_icon="💹", layout="wide")
    st.title("FinChat")
    st.caption("Ask questions about the provided financial news dataset.")

    @st.cache_resource(show_spinner=False)
    def load_service(cache_token: tuple[str, ...]) -> FinChatService:
        return FinChatService.from_path()

    service = load_service(_service_cache_token())
    runtime_llm_client = _render_sidebar(st, service)
    _initialize_session_state(st)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(_escape_currency_markdown(message["content"]))
            if message["role"] == "assistant":
                _render_sources(st, message["sources"])

    prompt = st.chat_input("Ask about recent financial news in the dataset")
    pending_prompt = st.session_state.pop("pending_prompt", None)
    active_prompt = prompt or pending_prompt

    if active_prompt:
        _process_prompt(st, service, active_prompt, llm_client=runtime_llm_client)


def _render_sidebar(st, service: FinChatService) -> Optional[LLMClient]:
    with st.sidebar:
        runtime_llm_client = _render_session_llm_controls(st)
        st.divider()
        st.subheader("Dataset")
        st.write(f"Tickers: {', '.join(service.available_tickers)}")
        st.write(
            "Answer mode: "
            + (
                "Session LLM synthesis"
                if runtime_llm_client is not None
                else ("LLM synthesis" if service.default_mode == "llm" else "Deterministic local summary")
            )
        )
        st.write(f"Retrieval: {service.retrieval_mode}")
        st.divider()
        st.subheader("Sample Questions")
        for index, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, key=f"sample-{index}", use_container_width=True):
                st.session_state.pending_prompt = question
    return runtime_llm_client


def _render_session_llm_controls(st) -> Optional[LLMClient]:
    st.subheader("Session LLM")
    st.caption("Paste a key here to use an LLM for this browser session only. Nothing is written to `.env`.")

    provider_label = st.selectbox(
        "Provider",
        tuple(SESSION_LLM_OPTIONS),
        key="session_llm_provider",
    )
    provider_spec = SESSION_LLM_OPTIONS[provider_label]
    api_key = st.text_input(
        "API key",
        type="password",
        key="session_llm_api_key",
    ).strip()
    model = st.text_input(
        "Model override",
        key="session_llm_model",
        placeholder=provider_spec["default_model"],
    ).strip()

    base_url = ""
    if provider_spec["requires_base_url"]:
        base_url = st.text_input(
            "Base URL",
            key="session_llm_base_url",
            placeholder="https://your-provider.example.com/v1",
        ).strip()

    if provider_spec["provider"] is None:
        st.caption("Local summary mode is active.")
        return None

    if not api_key:
        st.caption("Add a key to enable session-only LLM synthesis.")
        return None

    if provider_spec["requires_base_url"] and not base_url:
        st.caption("Add a base URL to use an OpenAI-compatible provider.")
        return None

    config = LLMConfig(
        provider=provider_spec["provider"],
        api_key=api_key,
        model=model or provider_spec["default_model"],
        base_url=base_url or None,
    )
    client = build_llm_client(config)
    if client is None:
        st.caption("The selected provider client is unavailable in this environment.")
        return None

    st.caption("Session key is active. It lives only in Streamlit session state.")
    return client


def _initialize_session_state(st) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _process_prompt(
    st,
    service: FinChatService,
    prompt: str,
    *,
    llm_client: Optional[LLMClient] = None,
) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(_escape_currency_markdown(prompt))

    with st.chat_message("assistant"):
        with st.spinner("Reviewing the dataset..."):
            result = service.answer(prompt, llm_client=llm_client)
        st.markdown(_escape_currency_markdown(result.answer_text))
        serialized_sources = _serialize_sources(result.sources)
        _render_sources(st, serialized_sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer_text,
            "sources": serialized_sources,
        }
    )


def _serialize_sources(sources) -> List[Dict[str, str]]:
    serialized = []
    for source in sources:
        serialized.append(
            {
                "ticker": source.article.ticker,
                "title": source.article.title,
                "link": source.article.link,
                "excerpt": source.excerpt,
                "score": f"{source.score:.3f}",
            }
        )
    return serialized


def _render_sources(st, sources: List[Dict[str, str]]) -> None:
    if not sources:
        return

    with st.expander("Sources", expanded=False):
        for source in sources:
            safe_title = _escape_currency_markdown(source["title"])
            st.markdown(f"**{source['ticker']}**: [{safe_title}]({source['link']})")
            st.caption(f"Retrieval score: {source['score']}")
            st.markdown(_escape_currency_markdown(source["excerpt"]))


def _escape_currency_markdown(text: str) -> str:
    return text.replace("$", r"\$")


def _service_cache_token() -> tuple[str, ...]:
    repo_root = Path(__file__).resolve().parents[2]
    watched_paths = [
        repo_root / "data" / "stock_news.json",
        *sorted((repo_root / "src" / "finchat").glob("*.py")),
    ]
    token_parts = []
    for path in watched_paths:
        stat = path.stat()
        token_parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return tuple(token_parts)
