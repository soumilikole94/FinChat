"""Provider-specific LLM adapters used only after local retrieval selects context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
import os

from .constants import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    INSUFFICIENT_SUPPORT_MESSAGE,
)
from .models import RetrievedContext


LLM_SYSTEM_PROMPT = (
    "You answer questions about financial news using only the supplied context. "
    "Do not use outside knowledge, common knowledge, assumptions, or prior facts. "
    "Synthesize across sources when relevant. "
    "If the supplied context does not explicitly support the answer, respond exactly with: "
    f"\"{INSUFFICIENT_SUPPORT_MESSAGE}\""
)

PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "compatible": "openai",
    "deepseek": "openai",
    "fireworks": "openai",
    "gemini": "gemini",
    "google": "gemini",
    "groq": "openai",
    "mistral": "openai",
    "openai": "openai",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "openrouter": "openai",
    "perplexity": "openai",
    "together": "openai",
}

PROVIDER_DEFAULT_MODELS = {
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
}


class LLMClient(Protocol):
    def generate_answer(self, query: str, contexts: Sequence[RetrievedContext]) -> str:
        ...


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    model: str
    base_url: Optional[str] = None


class OpenAIContextClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        base_url: Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_answer(self, query: str, contexts: Sequence[RetrievedContext]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": build_context_prompt(query, contexts)},
            ],
        )
        message = response.choices[0].message.content or ""
        return message.strip()


class AnthropicContextClient:
    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_answer(self, query: str, contexts: Sequence[RetrievedContext]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=700,
            temperature=0.2,
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_context_prompt(query, contexts)}],
        )
        return join_text_blocks(getattr(response, "content", []))


class GeminiContextClient:
    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL) -> None:
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_answer(self, query: str, contexts: Sequence[RetrievedContext]) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{LLM_SYSTEM_PROMPT}\n\n{build_context_prompt(query, contexts)}",
        )
        text = getattr(response, "text", "") or ""
        if text.strip():
            return text.strip()
        return extract_gemini_text(response)


def resolve_llm_config() -> Optional[LLMConfig]:
    explicit_provider = normalize_provider(os.getenv("LLM_PROVIDER"))
    if explicit_provider:
        return build_provider_config(explicit_provider, prefer_generic=True)

    for provider in ("openai", "anthropic", "gemini"):
        config = build_provider_config(provider, prefer_generic=False, require_provider_specific_key=True)
        if config is not None:
            return config

    generic_api_key = os.getenv("LLM_API_KEY")
    if not generic_api_key:
        return None

    return LLMConfig(
        provider="openai",
        api_key=generic_api_key,
        model=os.getenv("LLM_MODEL", DEFAULT_OPENAI_MODEL),
        base_url=os.getenv("LLM_BASE_URL"),
    )


def build_default_llm_client() -> Optional[LLMClient]:
    config = resolve_llm_config()
    return build_llm_client(config)


def build_llm_client(config: Optional[LLMConfig]) -> Optional[LLMClient]:
    if config is None:
        return None

    try:
        if config.provider == "anthropic":
            return AnthropicContextClient(api_key=config.api_key, model=config.model)
        if config.provider == "gemini":
            return GeminiContextClient(api_key=config.api_key, model=config.model)
        return OpenAIContextClient(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
        )
    except ImportError:
        return None


def build_context_prompt(query: str, contexts: Sequence[RetrievedContext]) -> str:
    context_sections = []
    for index, context in enumerate(contexts[:4], start=1):
        context_sections.append(
            (
                f"Source {index}\n"
                f"Ticker: {context.article.ticker}\n"
                f"Title: {context.article.title}\n"
                f"Link: {context.article.link}\n"
                f"Excerpt: {context.excerpt}\n"
            )
        )

    joined_context = "\n\n".join(context_sections)
    return (
        f"Question: {query}\n\n"
        "Answer only from the supplied sources. "
        "Do not rely on outside knowledge. "
        "If the sources are not enough, respond exactly with: "
        f"\"{INSUFFICIENT_SUPPORT_MESSAGE}\"\n\n"
        f"{joined_context}"
    )


def normalize_provider(provider: Optional[str]) -> Optional[str]:
    if not provider:
        return None
    return PROVIDER_ALIASES.get(provider.strip().lower())


def build_provider_config(
    provider: str,
    *,
    prefer_generic: bool,
    require_provider_specific_key: bool = False,
) -> Optional[LLMConfig]:
    provider_key = provider_specific_api_key(provider)
    if require_provider_specific_key and not provider_key:
        return None

    api_key = os.getenv("LLM_API_KEY") or provider_key if prefer_generic else provider_key
    if not api_key:
        return None

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model=provider_model(provider, prefer_generic=prefer_generic),
        base_url=provider_base_url(provider, prefer_generic=prefer_generic),
    )


def provider_specific_api_key(provider: str) -> Optional[str]:
    env_names = {
        "anthropic": ("ANTHROPIC_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "openai": ("OPENAI_API_KEY",),
    }[provider]
    return first_env(*env_names)


def provider_model(provider: str, *, prefer_generic: bool) -> str:
    env_name = {
        "anthropic": "ANTHROPIC_MODEL",
        "gemini": "GEMINI_MODEL",
        "openai": "OPENAI_MODEL",
    }[provider]
    if prefer_generic:
        return first_env("LLM_MODEL", env_name) or PROVIDER_DEFAULT_MODELS[provider]

    if provider == "gemini":
        return first_env("GEMINI_MODEL", "GOOGLE_MODEL", "LLM_MODEL") or PROVIDER_DEFAULT_MODELS[provider]

    return first_env(env_name, "LLM_MODEL") or PROVIDER_DEFAULT_MODELS[provider]


def provider_base_url(provider: str, *, prefer_generic: bool) -> Optional[str]:
    if provider != "openai":
        return None
    if prefer_generic:
        return first_env("LLM_BASE_URL", "OPENAI_BASE_URL")
    return first_env("OPENAI_BASE_URL", "LLM_BASE_URL")


def first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def join_text_blocks(blocks: Sequence[object]) -> str:
    parts = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text:
            parts.append(text.strip())
    return "\n".join(part for part in parts if part).strip()


def extract_gemini_text(response: object) -> str:
    candidates = getattr(response, "candidates", []) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        response_parts = getattr(content, "parts", []) if content is not None else []
        for part in response_parts:
            text = getattr(part, "text", None)
            if text:
                parts.append(text.strip())
    return "\n".join(part for part in parts if part).strip()
