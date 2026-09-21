from __future__ import annotations

import json
import re

from symptoms_recognizer.llm.base import LLMInvalidResponseError

_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(response_text: str) -> dict:
    """Pull a single JSON object out of a model reply.

    Models wrap JSON in prose or code fences often enough that this has to be
    tolerant, but a reply with no usable object raises rather than returning
    an empty result: the caller cannot otherwise tell "nothing found" from
    "the model did not answer".
    """
    if not response_text or not response_text.strip():
        raise LLMInvalidResponseError("Model returned an empty response")

    candidate = _FENCE_RE.sub("", response_text).strip()

    match = _OBJECT_RE.search(candidate)
    if not match:
        raise LLMInvalidResponseError(
            f"No JSON object in model response: {_preview(response_text)}"
        )

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMInvalidResponseError(
            f"Malformed JSON in model response: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMInvalidResponseError(
            f"Expected a JSON object, got {type(parsed).__name__}"
        )

    return parsed


def _preview(text: str, limit: int = 200) -> str:
    """A short, single-line excerpt for error messages.

    Kept deliberately short: a model reply echoes clinical text, so only
    enough to identify the failure belongs in a log line.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."
