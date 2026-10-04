"""Independent LLM calls run at once and are joined in order; nothing else changes.

The calls an extraction makes, and their prompts, must be exactly the ones it
made before: only how many wait at the same time changes. None of these load
the encoder or reach a provider.
"""
from __future__ import annotations

import threading
import time

import pytest
import torch

import symptoms_recognizer.concurrency as concurrency
import symptoms_recognizer.llm.claude_cli_provider as cli_module
import symptoms_recognizer.mapper.mapper as mapper_module
from symptoms_recognizer.llm.base import LLMProvider
from symptoms_recognizer.llm.caching import RateLimitedProvider
from symptoms_recognizer.llm.factory import build_provider
from symptoms_recognizer.mapper.mapper import PhenotypeOntologyMapper
from symptoms_recognizer.text_parser.sentences_parser import SentencesParser


class SlowProvider(LLMProvider):
    """Answers after a fixed delay and records how many calls overlapped."""

    def __init__(self, delay: float = 0.2, reply: str = '{"hpo_code": "HP:0000001"}'):
        super().__init__(model="slow")
        self.delay = delay
        self.reply = reply
        self.prompts: list[str] = []
        self.in_flight = 0
        self.peak = 0
        self._lock = threading.Lock()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        with self._lock:
            self.prompts.append(user_prompt)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        time.sleep(self.delay)
        with self._lock:
            self.in_flight -= 1
        return self.reply


def make_mapper(provider):
    mapper = PhenotypeOntologyMapper.__new__(PhenotypeOntologyMapper)
    mapper.api_provider = "stub"
    mapper.api_model_name = "stub"
    mapper.provider = provider
    mapper.prompt = None
    mapper.top_k = 3
    mapper.last_mapping_logs = []
    return mapper


def entries(n):
    return [(i, f"síntoma {i}", f"contexto {i}", "1. Código: HP:0000001 | Nombre: x\n") for i in range(n)]


def test_threads_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("RECOGNIZER_LLM_THREADS", raising=False)
    assert concurrency.llm_threads() == concurrency.DEFAULT_LLM_THREADS
    monkeypatch.setenv("RECOGNIZER_LLM_THREADS", "3")
    assert concurrency.llm_threads() == 3
    monkeypatch.setenv("RECOGNIZER_LLM_THREADS", "0")
    with pytest.raises(ValueError):
        concurrency.llm_threads()


def test_map_in_order_keeps_the_order_of_its_inputs(monkeypatch):
    monkeypatch.setenv("RECOGNIZER_LLM_THREADS", "8")

    def slow_echo(x):
        time.sleep(0.05 * (5 - x))  # later inputs finish first
        return x

    assert concurrency.map_in_order(slow_echo, range(5)) == [0, 1, 2, 3, 4]


def test_every_mention_is_mapped_at_once_up_to_the_ceiling(monkeypatch):
    monkeypatch.setenv("RECOGNIZER_LLM_THREADS", "16")
    provider = SlowProvider(delay=0.3)
    mapper = make_mapper(provider)
    mapped, logs = ["None"] * 12, [None] * 12

    started = time.monotonic()
    mapper._resolve_codes(entries(12), mapped, logs)
    elapsed = time.monotonic() - started

    assert provider.peak == 12  # one wave, where 5 threads made three
    assert elapsed < 0.3 * 2
    assert len(provider.prompts) == 12
    assert mapped == ["HP:0000001"] * 12


def test_the_ceiling_holds(monkeypatch):
    monkeypatch.setenv("RECOGNIZER_LLM_THREADS", "4")
    provider = SlowProvider(delay=0.05)
    mapper_module_mapper = make_mapper(provider)
    mapper_module_mapper._resolve_codes(entries(10), ["None"] * 10, [None] * 10)
    assert provider.peak <= 4


def test_the_prompts_are_the_ones_a_single_thread_sends(monkeypatch):
    sequential, parallel = SlowProvider(delay=0), SlowProvider(delay=0.01)
    monkeypatch.setenv("RECOGNIZER_LLM_THREADS", "1")
    make_mapper(sequential)._resolve_codes(entries(7), ["None"] * 7, [None] * 7)
    monkeypatch.setenv("RECOGNIZER_LLM_THREADS", "16")
    make_mapper(parallel)._resolve_codes(entries(7), ["None"] * 7, [None] * 7)
    assert sorted(parallel.prompts) == sorted(sequential.prompts)
    assert len(parallel.prompts) == len(sequential.prompts) == 7


def test_sentences_are_read_at_once_and_joined_in_order(monkeypatch):
    monkeypatch.setenv("RECOGNIZER_LLM_THREADS", "8")
    seen_together = []
    lock = threading.Lock()
    in_flight = [0]

    class Detector:
        def detect_phenotypes(self, sentence):
            with lock:
                in_flight[0] += 1
                seen_together.append(in_flight[0])
            time.sleep(0.1)
            with lock:
                in_flight[0] -= 1
            return [(sentence.split()[0], sentence)]

    text = "Uno tiene fiebre. Dos tiene tos. Tres tiene dolor. Cuatro tiene vómitos."
    result = SentencesParser().apply(text, Detector())
    assert [mention for mention, _ in result] == ["Uno", "Dos", "Tres", "Cuatro"]
    assert max(seen_together) > 1


def test_hpo_vectors_are_loaded_once(monkeypatch):
    loads = []

    def fake_batches(self):
        loads.append(1)
        yield {"HP:0000001": {"vector": torch.ones(1, 4), "name": "uno"}}

    monkeypatch.setattr(PhenotypeOntologyMapper, "_get_codes_batch", fake_batches)
    mapper = make_mapper(None)
    mapper.pytorch_device = torch.device("cpu")
    mapper._hpo_batches = None
    mapper._hpo_batches_lock = threading.Lock()
    first = mapper._hpo_vectors()
    second = mapper._hpo_vectors()
    assert first is second and len(loads) == 1
    codes, names, vectors = first[0]
    assert codes == ["HP:0000001"] and names == ["uno"] and tuple(vectors.shape) == (1, 4)


def test_the_burst_can_match_the_fan_out(monkeypatch):
    monkeypatch.setenv("LLM_REQUESTS_BURST", "20")
    provider = build_provider(provider="claude-cli", cache=False, requests_per_minute=60)
    assert isinstance(provider, RateLimitedProvider) and provider.burst == 20
    monkeypatch.delenv("LLM_REQUESTS_BURST")
    assert build_provider(provider="claude-cli", cache=False, requests_per_minute=60).burst == 10


def test_the_cli_runs_without_session_files_or_nonessential_traffic(monkeypatch):
    calls = {}

    def fake_run_cli(command, timeout, label, expect_stdout=True, env=None):
        calls["command"], calls["env"] = command, env
        return "{}"

    monkeypatch.setattr(cli_module, "run_cli", fake_run_cli)
    monkeypatch.setattr(cli_module, "require_binary", lambda binary, hint: binary)
    cli_module.ClaudeCLIProvider(model="claude-opus-5").complete("sistema", "usuario")
    assert calls["command"][:5] == ["claude", "-p", "usuario", "--model", "claude-opus-5"]
    assert "--no-session-persistence" in calls["command"]
    assert calls["command"][-2:] == ["--append-system-prompt", "sistema"]
    assert calls["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
