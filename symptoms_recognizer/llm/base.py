from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(Exception):
    """Base class for every failure raised by a provider.

    Callers are expected to let this propagate. Returning an empty phenotype
    list when the model was never reached is indistinguishable from "this
    patient has no symptoms", which is not a safe answer for a clinical tool.
    """


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""


class LLMInvalidResponseError(LLMError):
    """The provider answered, but the payload could not be used."""


class LLMProvider(ABC):
    """A text-in / text-out chat model.

    Both pipeline stages need exactly this: a system prompt describing the
    task, a user prompt carrying the clinical text, and the model's reply as a
    string. Keeping the interface this narrow is what lets the benchmark run on
    a subscription-backed CLI while production runs against a hosted API.
    """

    def __init__(self, model: str, timeout: int = 120):
        self.model = model
        self.timeout = timeout

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's reply, or raise an LLMError subclass."""

    @property
    def name(self) -> str:
        return f"{type(self).__name__}({self.model})"
