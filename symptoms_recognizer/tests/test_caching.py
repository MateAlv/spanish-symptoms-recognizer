"""Offline tests for the caching and rate-limiting provider wrappers."""
from __future__ import annotations

import threading
import time

import pytest

from symptoms_recognizer.llm.base import LLMError, LLMProvider
from symptoms_recognizer.llm.caching import (
    CachingProvider,
    RateLimitedProvider,
    RateLimitError,
    cache_key,
)


class CountingProvider(LLMProvider):
    def __init__(self, reply: str = "ok", delay: float = 0.0, error: Exception | None = None):
        super().__init__(model="counting")
        self.reply = reply
        self.delay = delay
        self.error = error
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        with self._lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return f"{self.reply}:{user_prompt}"


class TestCacheKey:
    def test_same_inputs_same_key(self):
        assert cache_key("m", "s", "u") == cache_key("m", "s", "u")

    @pytest.mark.parametrize(
        "a,b",
        [
            (("m", "s", "u"), ("m2", "s", "u")),
            (("m", "s", "u"), ("m", "s2", "u")),
            (("m", "s", "u"), ("m", "s", "u2")),
            # Field boundaries must matter, or concatenation would collide.
            (("m", "ab", "c"), ("m", "a", "bc")),
        ],
    )
    def test_different_inputs_differ(self, a, b):
        assert cache_key(*a) != cache_key(*b)

    def test_key_does_not_leak_clinical_text(self):
        key = cache_key("m", "sys", "Paciente con convulsiones")
        assert "convulsiones" not in key
        assert "Paciente" not in key


class TestCachingProvider:
    def test_repeated_call_hits_cache(self):
        inner = CountingProvider()
        provider = CachingProvider(inner)

        first = provider.complete("s", "u")
        second = provider.complete("s", "u")

        assert first == second
        assert inner.calls == 1
        assert (provider.hits, provider.misses) == (1, 1)

    def test_different_prompts_are_separate_entries(self):
        inner = CountingProvider()
        provider = CachingProvider(inner)
        provider.complete("s", "u1")
        provider.complete("s", "u2")
        assert inner.calls == 2

    def test_expired_entry_is_refetched(self):
        inner = CountingProvider()
        provider = CachingProvider(inner, ttl_seconds=0.05)
        provider.complete("s", "u")
        time.sleep(0.08)
        provider.complete("s", "u")
        assert inner.calls == 2

    def test_eviction_is_bounded(self):
        inner = CountingProvider()
        provider = CachingProvider(inner, max_entries=3)
        for i in range(5):
            provider.complete("s", f"u{i}")
        assert len(provider._entries) == 3

    def test_least_recently_used_is_evicted_first(self):
        inner = CountingProvider()
        provider = CachingProvider(inner, max_entries=2)
        provider.complete("s", "keep")
        provider.complete("s", "drop")
        provider.complete("s", "keep")   # refreshes recency
        provider.complete("s", "new")    # evicts "drop"

        calls_before = inner.calls
        provider.complete("s", "keep")
        assert inner.calls == calls_before  # still cached

    def test_failures_are_not_cached(self):
        # A cached outage would keep failing long after the provider recovers.
        inner = CountingProvider(error=LLMError("down"))
        provider = CachingProvider(inner)
        for _ in range(2):
            with pytest.raises(LLMError):
                provider.complete("s", "u")
        assert inner.calls == 2

    def test_concurrent_callers_get_consistent_results(self):
        inner = CountingProvider(delay=0.02)
        provider = CachingProvider(inner)
        results = []

        def call():
            results.append(provider.complete("s", "u"))

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1
        assert len(results) == 8

    def test_name_reports_the_wrapped_provider(self):
        assert "CountingProvider" in CachingProvider(CountingProvider()).name


class TestRateLimitedProvider:
    def test_calls_within_burst_are_not_delayed(self):
        provider = RateLimitedProvider(CountingProvider(), requests_per_minute=600, burst=5)
        started = time.monotonic()
        for _ in range(5):
            provider.complete("s", "u")
        assert time.monotonic() - started < 0.2

    def test_exceeding_the_budget_waits(self):
        # 600/min = one per 0.1s; a 6th call past a burst of 5 must wait.
        provider = RateLimitedProvider(CountingProvider(), requests_per_minute=600, burst=5)
        for _ in range(5):
            provider.complete("s", "u")

        started = time.monotonic()
        provider.complete("s", "u")
        assert time.monotonic() - started >= 0.05

    def test_budget_exhaustion_raises_rather_than_hanging(self):
        provider = RateLimitedProvider(
            CountingProvider(), requests_per_minute=6, burst=1, max_wait_seconds=0.1
        )
        provider.complete("s", "u")
        with pytest.raises(RateLimitError):
            provider.complete("s", "u")

    def test_rate_limit_error_is_an_llm_error(self):
        # Callers already handle LLMError, so this must not escape as a new type.
        assert issubclass(RateLimitError, LLMError)

    def test_tokens_refill_over_time(self):
        provider = RateLimitedProvider(CountingProvider(), requests_per_minute=600, burst=1)
        provider.complete("s", "u")
        time.sleep(0.15)
        started = time.monotonic()
        provider.complete("s", "u")
        assert time.monotonic() - started < 0.05

    def test_invalid_rate_is_rejected(self):
        with pytest.raises(ValueError):
            RateLimitedProvider(CountingProvider(), requests_per_minute=0)

    def test_concurrent_callers_do_not_exceed_the_bucket(self):
        inner = CountingProvider()
        provider = RateLimitedProvider(
            inner, requests_per_minute=60, burst=4, max_wait_seconds=0.05
        )
        allowed = []

        def call():
            try:
                provider.complete("s", "u")
                allowed.append(1)
            except RateLimitError:
                pass

        threads = [threading.Thread(target=call) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The bucket holds 4; refill at 60/min is ~0.07 tokens in 70ms.
        assert len(allowed) <= 5
        assert inner.calls == len(allowed)


class TestStackedWrappers:
    def test_cache_hit_does_not_spend_a_rate_limit_token(self):
        inner = CountingProvider()
        provider = CachingProvider(
            RateLimitedProvider(inner, requests_per_minute=60, burst=1, max_wait_seconds=0.05)
        )
        first = provider.complete("s", "u")
        second = provider.complete("s", "u")  # would be throttled if it reached the bucket

        assert first == second
        assert inner.calls == 1
