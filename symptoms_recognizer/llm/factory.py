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


def build_provider(
    provider: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> LLMProvider:
    """Build the configured LLM provider.

    Falls back to the `LLM_PROVIDER` / `LLM_MODEL` environment variables so a
    benchmark can be pointed at codex locally and at a hosted API in
    production without touching the pipeline code.
    """
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
