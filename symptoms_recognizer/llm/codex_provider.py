from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time

from symptoms_recognizer.llm.base import LLMError, LLMProvider, LLMTimeoutError

logger = logging.getLogger(__name__)

DEFAULT_CODEX_BINARY = "codex"
DEFAULT_MODEL = "gpt-5.6-terra"


class CodexProvider(LLMProvider):
    """Talks to the `codex` CLI, billed against an interactive subscription.

    Intended for local benchmarking, where per-token billing is not wanted.
    Two limits make this unsuitable for production, both verified on TARS with
    codex-cli 0.155.1:

    - `claude-opus-5` and `gpt-4o-mini` are refused on a ChatGPT account, so the
      exact configuration behind the module's published benchmark cannot be
      reproduced here.
    - Authentication is an interactive login on the host, which a container has
      no way to provide.

    Production is expected to use an API-backed provider instead; that is a
    change of configuration, not of calling code.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: int = 180,
        binary: str | None = None,
    ):
        super().__init__(model=model, timeout=timeout)
        self.binary = binary or os.environ.get("CODEX_BINARY", DEFAULT_CODEX_BINARY)

        if shutil.which(self.binary) is None:
            raise LLMError(
                f"codex binary {self.binary!r} not found on PATH. "
                "Install codex-cli or set CODEX_BINARY."
            )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # codex exec takes a single prompt rather than a system/user pair.
        prompt = f"{system_prompt}\n\n{user_prompt}"

        # The output file must live outside the repo tree, and the directory
        # goes away even if the call raises.
        with tempfile.TemporaryDirectory(prefix="codex-llm-") as tmpdir:
            out_path = os.path.join(tmpdir, "out.txt")
            command = [
                self.binary,
                "exec",
                "--skip-git-repo-check",
                "-m",
                self.model,
                "-o",
                out_path,
                prompt,
            ]

            started = time.time()
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,  # codex exec hangs forever otherwise
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMTimeoutError(
                    f"codex did not answer within {self.timeout}s (model={self.model})"
                ) from exc

            elapsed = time.time() - started

            if completed.returncode != 0:
                raise LLMError(
                    f"codex exited {completed.returncode} (model={self.model}): "
                    f"{_tail(completed.stderr or completed.stdout)}"
                )

            try:
                with open(out_path, encoding="utf-8") as handle:
                    output = handle.read().strip()
            except OSError as exc:
                raise LLMError(f"codex wrote no output file (model={self.model})") from exc

        if not output:
            raise LLMError(f"codex returned an empty response (model={self.model})")

        # The prompt carries clinical text and is deliberately never logged.
        logger.info(
            "codex model=%s elapsed=%.1fs chars=%d", self.model, elapsed, len(output)
        )
        return output


def _tail(text: str | None, limit: int = 300) -> str:
    if not text:
        return "(no stderr)"
    collapsed = " ".join(text.split())
    return collapsed[-limit:]
