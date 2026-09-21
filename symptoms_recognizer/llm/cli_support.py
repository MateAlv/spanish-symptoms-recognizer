"""Shared mechanics for providers that shell out to a CLI.

Both CLI-backed providers have the same three hazards: a subprocess that
blocks forever on stdin, clinical text that must never reach a shell, and a
wedged process that must not hang the caller. They are handled once here.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time

from symptoms_recognizer.llm.base import LLMError, LLMTimeoutError

logger = logging.getLogger(__name__)


def require_binary(binary: str, hint: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        raise LLMError(f"{binary!r} not found on PATH. {hint}")
    return resolved


def run_cli(
    command: list[str], timeout: int, label: str, expect_stdout: bool = True
) -> str:
    """Run a CLI command and return its stdout.

    The prompt is passed as an argument-list element and never through a
    shell, so clinical text containing quotes or `$` cannot alter the command.

    `expect_stdout=False` suits a CLI that writes its reply to a file, where
    empty stdout is normal rather than a failure.
    """
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,  # these CLIs block forever on an open stdin
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMTimeoutError(f"{label} did not answer within {timeout}s") from exc

    elapsed = time.time() - started

    if completed.returncode != 0:
        raise LLMError(
            f"{label} exited {completed.returncode}: "
            f"{tail(completed.stderr or completed.stdout)}"
        )

    output = (completed.stdout or "").strip()
    if expect_stdout and not output:
        raise LLMError(f"{label} returned an empty response")

    # The prompt carries clinical text and is deliberately never logged.
    logger.info("%s elapsed=%.1fs chars=%d", label, elapsed, len(output))
    return output


def tail(text: str | None, limit: int = 300) -> str:
    if not text:
        return "(no output)"
    collapsed = " ".join(text.split())
    return collapsed[-limit:]
