from __future__ import annotations

import os

from symptoms_recognizer.llm.base import LLMError, LLMProvider

CODEX_PROVIDER = "codex"
CLAUDE_CLI_PROVIDER = "claude-cli"
OPENAI_PROVIDER = "openai"
ANTHROPIC_PROVIDER = "anthropic"
GEMINI_PROVIDER = "gemini"

KNOWN_PROVIDERS = (
    CODEX_PROVIDER,
    CLAUDE_CLI_PROVIDER,
    OPENAI_PROVIDER,
    ANTHROPIC_PROVIDER,
    GEMINI_PROVIDER,
)

DEFAULT_TIMEOUT_SECONDS = 180

# Both CLI providers are subscription-throttled rather than billed per token,
# so a default ceiling is safer than none. Overridden by LLM_REQUESTS_PER_MINUTE.
DEFAULT_REQUESTS_PER_MINUTE = 60.0


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def build_provider(
    provider: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
    cache: bool | None = None,
    requests_per_minute: float | None = None,
) -> LLMProvider:
    """Build the configured LLM provider.

    Falls back to the `LLM_PROVIDER` / `LLM_MODEL` environment variables so a
    benchmark can be pointed at codex locally and at a hosted API in
    production without touching the pipeline code.
    """
    base = _build_base_provider(provider, model, timeout)

    # Rate limiting goes on first, so the cache can answer without spending a
    # token from the bucket.
    rpm = (
        requests_per_minute
        if requests_per_minute is not None
        else float(os.environ.get("LLM_REQUESTS_PER_MINUTE", DEFAULT_REQUESTS_PER_MINUTE))
    )
    if rpm > 0:
        from symptoms_recognizer.llm.caching import RateLimitedProvider

        base = RateLimitedProvider(base, requests_per_minute=rpm)

    if cache if cache is not None else _env_flag("LLM_CACHE_ENABLED", True):
        from symptoms_recognizer.llm.caching import CachingProvider

        base = CachingProvider(base)

    return base


def _build_base_provider(
    provider: str | None, model: str | None, timeout: int | None
) -> LLMProvider:
    provider = (
        provider or os.environ.get("LLM_PROVIDER") or CLAUDE_CLI_PROVIDER
    ).lower()
    model = model or os.environ.get("LLM_MODEL")
    timeout = timeout or int(
        os.environ.get("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    )

    if provider == CLAUDE_CLI_PROVIDER:
        from symptoms_recognizer.llm.claude_cli_provider import (
            DEFAULT_MODEL,
            ClaudeCLIProvider,
        )

        return ClaudeCLIProvider(model=model or DEFAULT_MODEL, timeout=timeout)

    if provider == CODEX_PROVIDER:
        from symptoms_recognizer.llm.codex_provider import (
            DEFAULT_MODEL,
            CodexProvider,
        )

        return CodexProvider(model=model or DEFAULT_MODEL, timeout=timeout)

    if provider not in KNOWN_PROVIDERS:
        raise LLMError(
            f"Unknown LLM provider {provider!r}. Expected one of: "
            f"{', '.join(KNOWN_PROVIDERS)}"
        )

    if not model:
        raise LLMError(f"A model name is required for provider {provider!r}")

    from symptoms_recognizer.llm import api_providers

    builders = {
        OPENAI_PROVIDER: api_providers.OpenAIProvider,
        ANTHROPIC_PROVIDER: api_providers.AnthropicProvider,
        GEMINI_PROVIDER: api_providers.GeminiProvider,
    }
    return builders[provider](model=model, timeout=timeout)
