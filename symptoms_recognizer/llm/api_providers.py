from __future__ import annotations

import logging
import time

from symptoms_recognizer.llm.base import LLMError, LLMProvider, LLMTimeoutError

logger = logging.getLogger(__name__)

# Some reasoning models reject sampling or token-budget parameters outright.
# Each provider therefore degrades through a list of keyword sets rather than
# failing on the first rejection.
_OPENAI_PARAM_FALLBACKS = (
    {"temperature": 0.0, "max_tokens": 2048},
    {"temperature": 0.0, "max_completion_tokens": 2048},
    {"max_completion_tokens": 2048},
    {},
)

_ANTHROPIC_PARAM_FALLBACKS = (
    {"temperature": 0.0, "max_tokens": 4096},
    {"max_tokens": 4096},
)


class _ApiProvider(LLMProvider):
    """Shared timing and logging for the hosted-API providers."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        started = time.time()
        try:
            output = self._invoke(system_prompt, user_prompt)
        except LLMError:
            raise
        except Exception as exc:
            if _looks_like_timeout(exc):
                raise LLMTimeoutError(
                    f"{self.name} timed out after {self.timeout}s"
                ) from exc
            raise LLMError(f"{self.name} call failed: {exc}") from exc

        if not output or not output.strip():
            raise LLMError(f"{self.name} returned an empty response")

        # Never log the prompt; it carries clinical text.
        logger.info(
            "llm provider=%s elapsed=%.1fs chars=%d",
            self.name,
            time.time() - started,
            len(output),
        )
        return output.strip()

    def _invoke(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIProvider(_ApiProvider):
    def __init__(self, model: str, timeout: int = 120):
        super().__init__(model=model, timeout=timeout)
        from openai import OpenAI

        self.client = OpenAI(timeout=timeout)

    def _invoke(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None

        for params in _OPENAI_PARAM_FALLBACKS:
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, **params
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                if not _is_bad_request(exc):
                    raise
                last_error = exc

        raise LLMError(f"{self.name} rejected every parameter set: {last_error}")


class AnthropicProvider(_ApiProvider):
    def __init__(self, model: str, timeout: int = 120):
        super().__init__(model=model, timeout=timeout)
        import anthropic

        self.client = anthropic.Anthropic(timeout=timeout)

    def _invoke(self, system_prompt: str, user_prompt: str) -> str:
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages = [{"role": "user", "content": user_prompt}]
        last_error: Exception | None = None

        for params in _ANTHROPIC_PARAM_FALLBACKS:
            try:
                response = self.client.messages.create(
                    model=self.model, system=system, messages=messages, **params
                )
                return "".join(
                    block.text for block in response.content if block.type == "text"
                )
            except Exception as exc:
                if not _is_bad_request(exc):
                    raise
                last_error = exc

        raise LLMError(f"{self.name} rejected every parameter set: {last_error}")


class GeminiProvider(_ApiProvider):
    def __init__(self, model: str, timeout: int = 120):
        super().__init__(model=model, timeout=timeout)
        from google import genai

        self.client = genai.Client()

    def _invoke(self, system_prompt: str, user_prompt: str) -> str:
        interaction = self.client.interactions.create(
            model=self.model,
            input=f"{system_prompt}\n\n{user_prompt}",
            config={"temperature": 0.0},
        )
        return interaction.output_text or ""


def _is_bad_request(exc: Exception) -> bool:
    """True when the model rejected the request shape rather than failing.

    Only these are worth retrying with fewer parameters; an auth error or an
    outage must surface immediately.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 400:
        return True
    return "unsupported" in str(exc).lower() or "unexpected keyword" in str(exc).lower()


def _looks_like_timeout(exc: Exception) -> bool:
    return "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()
