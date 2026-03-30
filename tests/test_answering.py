from pathlib import Path

import finchat.llm as llm_module
from finchat.answering import answer_question
from finchat.data import load_articles
from finchat.llm import LLMConfig, build_default_llm_client, build_llm_client, resolve_llm_config
from finchat.retrieval import build_index, retrieve
from finchat.service import FinChatService


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_news.json"


class StubLLMClient:
    def __init__(self) -> None:
        self.calls = []

    def generate_answer(self, query, contexts):
        self.calls.append((query, contexts))
        return "Stubbed LLM answer"


def clear_llm_env(monkeypatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GOOGLE_API_KEY",
        "GOOGLE_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_multi_source_synthesis_in_local_mode():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("Summarize the AI-related news across Microsoft and IBM.", index=index)
    result = answer_question("Summarize the AI-related news across Microsoft and IBM.", contexts)

    assert result.mode == "local"
    assert len(result.sources) >= 2
    assert len({source.article.ticker for source in result.sources}) >= 2
    assert result.answer_text
    assert not result.answer_text.startswith(
        "Based on the provided dataset, here are the main points from the retrieved article text"
    )
    assert "grounded only in the retrieved articles" not in result.answer_text.lower()


def test_unsupported_question_stays_grounded():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("What did JPMorgan say about new banking regulation?", index=index)
    result = answer_question("What did JPMorgan say about new banking regulation?", contexts)

    assert result.mode == "local"
    assert "couldn't find enough support" in result.answer_text.lower()


def test_ambiguous_apple_fruit_question_is_rejected_before_llm_use():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("Is Apple the company related to the fruit apple?", index=index)
    stub_client = StubLLMClient()

    result = answer_question("Is Apple the company related to the fruit apple?", contexts, llm_client=stub_client)

    assert result.mode == "local"
    assert "couldn't find enough support" in result.answer_text.lower()
    assert len(stub_client.calls) == 0


def test_llm_client_is_mockable():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("Summarize NVIDIA coverage.", index=index)
    stub_client = StubLLMClient()

    result = answer_question("Summarize NVIDIA coverage.", contexts, llm_client=stub_client)

    assert result.mode == "llm"
    assert result.answer_text == "Stubbed LLM answer"
    assert len(stub_client.calls) == 1


def test_service_initializes_without_api_key(monkeypatch):
    clear_llm_env(monkeypatch)
    service = FinChatService.from_path(DATA_PATH)
    result = service.answer("What is the latest Apple news in the dataset?")

    assert service.default_mode == "local"
    assert result.answer_text
    assert result.sources


def test_service_accepts_runtime_llm_override(monkeypatch):
    clear_llm_env(monkeypatch)
    service = FinChatService.from_path(DATA_PATH)
    stub_client = StubLLMClient()

    result = service.answer("Summarize NVIDIA coverage.", llm_client=stub_client)

    assert result.mode == "llm"
    assert result.answer_text == "Stubbed LLM answer"
    assert len(stub_client.calls) == 1


def test_resolve_llm_config_detects_openai_env(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    config = resolve_llm_config()

    assert config is not None
    assert config.provider == "openai"
    assert config.api_key == "openai-test-key"
    assert config.model == "gpt-test"


def test_resolve_llm_config_detects_anthropic_env(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")

    config = resolve_llm_config()

    assert config is not None
    assert config.provider == "anthropic"
    assert config.api_key == "anthropic-test-key"
    assert config.model == "claude-test"


def test_resolve_llm_config_detects_gemini_env(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "gemini-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")

    config = resolve_llm_config()

    assert config is not None
    assert config.provider == "gemini"
    assert config.api_key == "gemini-test-key"
    assert config.model == "gemini-test"


def test_build_default_llm_client_supports_openai_compatible_endpoint(monkeypatch):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_API_KEY", "generic-test-key")
    monkeypatch.setenv("LLM_MODEL", "llama-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")

    captured = {}

    class FakeOpenAIClient:
        def __init__(self, api_key, model, base_url=None):
            captured["api_key"] = api_key
            captured["model"] = model
            captured["base_url"] = base_url

    monkeypatch.setattr(llm_module, "OpenAIContextClient", FakeOpenAIClient)

    client = build_default_llm_client()

    assert isinstance(client, FakeOpenAIClient)
    assert captured == {
        "api_key": "generic-test-key",
        "model": "llama-test",
        "base_url": "https://example.test/v1",
    }


def test_build_llm_client_supports_explicit_runtime_config(monkeypatch):
    captured = {}

    class FakeOpenAIClient:
        def __init__(self, api_key, model, base_url=None):
            captured["api_key"] = api_key
            captured["model"] = model
            captured["base_url"] = base_url

    monkeypatch.setattr(llm_module, "OpenAIContextClient", FakeOpenAIClient)

    client = build_llm_client(
        LLMConfig(
            provider="openai",
            api_key="runtime-test-key",
            model="runtime-model",
            base_url="https://runtime.example/v1",
        )
    )

    assert isinstance(client, FakeOpenAIClient)
    assert captured == {
        "api_key": "runtime-test-key",
        "model": "runtime-model",
        "base_url": "https://runtime.example/v1",
    }


def test_local_summary_prefers_factual_sentences_over_clickbait_titles():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("What is the latest Apple news in the dataset?", index=index)
    result = answer_question("What is the latest Apple news in the dataset?", contexts)

    assert "most profitable tech stock" not in result.answer_text.lower()
    assert "should you buy apple stock" not in result.answer_text.lower()
    assert (
        "february 19" in result.answer_text.lower()
        or "feb. 19" in result.answer_text.lower()
        or "planning to launch a product" in result.answer_text.lower()
        or "apple intelligence" in result.answer_text.lower()
        or "in-house modem" in result.answer_text.lower()
    )


def test_apple_ai_china_query_stays_supported():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("What does the dataset say about Apple's AI plans in China?", index=index)
    result = answer_question("What does the dataset say about Apple's AI plans in China?", contexts)

    assert "couldn't find enough support" not in result.answer_text.lower()
    assert (
        "china" in result.answer_text.lower()
        or "alibaba" in result.answer_text.lower()
        or "baidu" in result.answer_text.lower()
    )


def test_local_summary_surfaces_intel_event_details():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("What is happening with Intel?", index=index)
    result = answer_question("What is happening with Intel?", contexts)

    assert "where will intel stock be in 1 year" not in result.answer_text.lower()
    assert "worth $167." not in result.answer_text.lower()
    assert (
        "broadcom" in result.answer_text.lower()
        or "tsmc" in result.answer_text.lower()
        or "split" in result.answer_text.lower()
    )


def test_local_summary_uses_article_text_for_nvidia_query():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("Summarize NVIDIA coverage.", index=index)
    result = answer_question("Summarize NVIDIA coverage.", contexts)

    assert "jim cramer on nvidia" not in result.answer_text.lower()
    assert "(nasdaq:nvda)" not in result.answer_text.lower()
    assert "in this article, we are going to take a look" not in result.answer_text.lower()
    assert "cal-maine foods" not in result.answer_text.lower()
    assert (
        "accelerated computing" in result.answer_text.lower()
        or "quiet period" in result.answer_text.lower()
        or "surged by 34%" in result.answer_text.lower()
        or "84% market" in result.answer_text.lower()
    )


def test_comparison_query_returns_both_companies_not_single_ticker_noise():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("is nvidia doing better than aapl?", index=index)
    result = answer_question("is nvidia doing better than aapl?", contexts)

    assert "nvidia" in result.answer_text.lower()
    assert "apple" in result.answer_text.lower()
    assert "apples-to-apples benchmark" in result.answer_text.lower()
    assert "apple (aapl): nvidia slashes" not in result.answer_text.lower()
    assert "apple (aapl): moreover, the model" not in result.answer_text.lower()
    assert (
        "apple's new device will also be its first device with an in-house cellular modem chip" in result.answer_text.lower()
        or "alibaba ai alliance" in result.answer_text.lower()
        or "china still presents a hurdle for apple" in result.answer_text.lower()
    )


def test_local_summary_can_return_up_to_four_bullets():
    index = build_index(load_articles(DATA_PATH))
    contexts = retrieve("What is happening with Intel?", index=index)
    result = answer_question("What is happening with Intel?", contexts)

    bullet_lines = [line for line in result.answer_text.splitlines() if line.startswith("- ")]
    assert len(bullet_lines) <= 4
    assert len(bullet_lines) >= 3
