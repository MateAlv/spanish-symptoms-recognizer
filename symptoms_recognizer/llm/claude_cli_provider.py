"""Provider backed by the `claude` CLI, billed against a Claude subscription."""
from __future__ import annotations

import os

from symptoms_recognizer.llm.base import LLMProvider
from symptoms_recognizer.llm.cli_support import require_binary, run_cli

DEFAULT_CLAUDE_BINARY = "claude"
DEFAULT_MODEL = "claude-opus-5"


class ClaudeCLIProvider(LLMProvider):
    """Talks to `claude -p`, billed against an interactive subscription.

    Preferred over the codex CLI for benchmarking, on three measurements taken
    on TARS: it reaches `claude-opus-5` (the model behind the module's
    published benchmark, which codex refuses on a ChatGPT account), it answered
    a 12-mention batch in 5.5s against codex's 15s, and it mapped that batch
    with no errors.

    It shares codex's production blocker: authentication is a credentials file
    belonging to a logged-in user, which a container on nodo0 does not have.
    Production is expected to use an API-backed provider, which is a change of
    configuration rather than of calling code.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: int = 180,
        binary: str | None = None,
    ):
        super().__init__(model=model, timeout=timeout)
        self.binary = binary or os.environ.get("CLAUDE_BINARY", DEFAULT_CLAUDE_BINARY)
        require_binary(
            self.binary,
            "Install the Claude Code CLI or set CLAUDE_BINARY.",
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # --no-session-persistence: each call is one-off, so writing its
        # session to disk is wasted work.
        command = [
            self.binary, "-p", user_prompt, "--model", self.model, "--no-session-persistence",
        ]

        # `claude -p` has no system-prompt slot of its own; appending keeps the
        # instructions ahead of the clinical text, as the API shape does.
        if system_prompt:
            command += ["--append-system-prompt", system_prompt]

        # Telemetry, error reporting and update checks add about a second to
        # every start of the CLI and nothing to the reply; the prompt and the
        # model are untouched.
        env = {**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
        return run_cli(command, timeout=self.timeout, label=f"claude({self.model})", env=env)
