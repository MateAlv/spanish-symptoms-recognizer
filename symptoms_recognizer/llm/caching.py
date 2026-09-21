"""Caching and rate limiting, applied at the provider seam.

Both wrap any LLMProvider, so the NER and mapping stages get them without
either knowing. A clinical history costs one LLM call per sentence plus one
per mention, so repeated work is common: the same history re-submitted, and
mentions like "fiebre" recurring across patients.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict

from symptoms_recognizer.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 2048
DEFAULT_TTL_SECONDS = 3600


class RateLimitError(LLMError):
    """The configured request budget is exhausted."""


def cache_key(model: str, system_prompt: str, user_prompt: str) -> str:
    """A stable key for one completion request.

    The prompts carry clinical text, so the key is a digest rather than the
    text itself: cache keys reach logs and metrics, and the text must not.
    """
    digest = hashlib.sha256()
    for part in (model, system_prompt, user_prompt):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")  # so ("ab","c") and ("a","bc") differ
    return digest.hexdigest()


class CachingProvider(LLMProvider):
    """Wraps a provider with an in-process TTL cache.

    Deliberately in-process rather than Redis: this module is a library, and
    the service embedding it already owns a Redis cache at the request level.
    Caching whole requests there is strictly better than caching fragments
    here, so this exists to spare repeated identical calls within a run --
    which is what a benchmark sweep and a re-submitted history both do.
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        super().__init__(model=provider.model, timeout=provider.timeout)
        self.provider = provider
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

        self._entries: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        key = cache_key(self.model, system_prompt, user_prompt)

        cached = self._get(key)
        if cached is not None:
            self.hits += 1
            logger.debug("llm cache hit model=%s", self.model)
            return cached

        self.misses += 1
        # Called outside the lock: a slow model call must not block other
        # threads, and the mapper fans out across a thread pool.
        response = self.provider.complete(system_prompt, user_prompt)
        self._put(key, response)
        return response

    def _get(self, key: str) -> str | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if expires_at <= now:
                del self._entries[key]
                return None

            self._entries.move_to_end(key)
            return value

    def _put(self, key: str, value: str) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic() + self.ttl_seconds, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    @property
    def name(self) -> str:
        return f"Caching({self.provider.name})"


class RateLimitedProvider(LLMProvider):
    """Wraps a provider with a token-bucket request limit.

    A subscription-backed CLI is throttled rather than billed per token, and
    exhausting it mid-run fails every later call -- as codex did here. A
    bucket refilling at a steady rate keeps a long run inside the budget, and
    the burst allowance keeps the mapper's thread-pool fan-out working.
    """

    def __init__(
        self,
        provider: LLMProvider,
        requests_per_minute: float = 60.0,
        burst: int | None = None,
        max_wait_seconds: float = 30.0,
    ):
        super().__init__(model=provider.model, timeout=provider.timeout)
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")

        self.provider = provider
        self.requests_per_minute = requests_per_minute
        self.burst = burst if burst is not None else max(1, int(requests_per_minute // 6))
        self.max_wait_seconds = max_wait_seconds

        self._tokens = float(self.burst)
        self._updated_at = time.monotonic()
        self._lock = threading.Lock()
        self.throttled = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self._acquire()
        return self.provider.complete(system_prompt, user_prompt)

    def _acquire(self) -> None:
        deadline = time.monotonic() + self.max_wait_seconds

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # Time until one token exists, computed under the lock.
                wait = (1.0 - self._tokens) * 60.0 / self.requests_per_minute

            if time.monotonic() + wait > deadline:
                raise RateLimitError(
                    f"rate limit of {self.requests_per_minute}/min exhausted; "
                    f"waited {self.max_wait_seconds}s for a slot"
                )

            self.throttled += 1
            time.sleep(min(wait, 0.5))

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated_at
        self._updated_at = now
        self._tokens = min(
            float(self.burst),
            self._tokens + elapsed * self.requests_per_minute / 60.0,
        )

    @property
    def name(self) -> str:
        return f"RateLimited({self.provider.name})"
