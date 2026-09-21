from __future__ import annotations

import os
import tempfile

from symptoms_recognizer.llm.base import LLMError, LLMProvider
from symptoms_recognizer.llm.cli_support import require_binary, run_cli

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
        require_binary(self.binary, "Install codex-cli or set CODEX_BINARY.")

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

            label = f"codex({self.model})"
            # codex writes the reply to -o rather than stdout, so its own
            # stdout is only useful for diagnosing a failure.
            run_cli(command, timeout=self.timeout, label=label, expect_stdout=False)

            try:
                with open(out_path, encoding="utf-8") as handle:
                    output = handle.read().strip()
            except OSError as exc:
                raise LLMError(f"{label} wrote no output file") from exc

        if not output:
            raise LLMError(f"{label} returned an empty response")

        return output
