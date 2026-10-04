"""How many LLM calls a single extraction may have in flight at once.

Every call an extraction makes is independent of the others in its stage --
each mention is mapped on its own, each sentence or chunk is read on its own
-- so they run concurrently and are joined in their original order. The
number of calls and their prompts do not change, only how many wait at once.

The ceiling is per extraction, set by RECOGNIZER_LLM_THREADS. Each in-flight
call of the claude CLI is a process of about 260 MB, so the right value
depends on the host: a big shared server can take 16, a 2 GB instance 2. The
default is the 5 the mapper always used, so a deployment that does not set it
behaves as before.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

DEFAULT_LLM_THREADS = 5

T = TypeVar("T")
R = TypeVar("R")


def llm_threads() -> int:
    raw = os.environ.get("RECOGNIZER_LLM_THREADS")
    if raw is None or not raw.strip():
        return DEFAULT_LLM_THREADS
    value = int(raw)
    if value < 1:
        raise ValueError(f"RECOGNIZER_LLM_THREADS must be at least 1, got {value}")
    return value


def map_in_order(fn: Callable[[T], R], items: Sequence[T] | Iterable[T]) -> list[R]:
    """`[fn(x) for x in items]`, with up to llm_threads() calls at once."""
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(llm_threads(), len(items))) as executor:
        return list(executor.map(fn, items))
