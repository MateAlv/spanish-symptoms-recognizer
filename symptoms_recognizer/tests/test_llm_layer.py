"""Offline tests for the provider layer and JSON parsing.

Nothing here reaches a network or a paid API, so the suite runs in CI and on a
machine with no credentials.
"""
from __future__ import annotations

import pytest

from symptoms_recognizer.llm.base import (
    LLMError,
    LLMInvalidResponseError,
    LLMProvider,
)
from symptoms_recognizer.llm.json_parsing import parse_json_object


class StubProvider(LLMProvider):
    """Returns canned replies, so pipeline behaviour is testable offline."""

    def __init__(self, replies: list[str] | None = None, error: Exception | None = None):
        super().__init__(model="stub")
        self.replies = list(replies or [])
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return self.replies.pop(0) if self.replies else ""


class TestParseJsonObject:
    def test_plain_object(self):
        assert parse_json_object('{"hpo_code": "HP:0001250"}') == {"hpo_code": "HP:0001250"}

    def test_fenced_object(self):
        reply = 'Mi analisis...\n```json\n{"hpo_code": "HP:0001250"}\n```'
        assert parse_json_object(reply) == {"hpo_code": "HP:0001250"}

    def test_object_after_prose(self):
        reply = 'El candidato 2 es el correcto porque encaja.\n{"hpo_code": "HP:0002315"}'
        assert parse_json_object(reply) == {"hpo_code": "HP:0002315"}

    @pytest.mark.parametrize(
        "reply",
        ["", "   ", "no hay JSON aqui", '{"hpo_code": ', "[1, 2, 3]"],
    )
    def test_unusable_replies_raise(self, reply):
        # The point of the layer: never silently degrade to an empty result.
        with pytest.raises(LLMInvalidResponseError):
            parse_json_object(reply)

    def test_error_preview_is_single_line_and_bounded(self):
        with pytest.raises(LLMInvalidResponseError) as excinfo:
            parse_json_object("linea1\n" + "x" * 500)
        message = str(excinfo.value)
        assert "\n" not in message
        assert len(message) < 400


class TestNerParsing:
    """Covers _parse_llm_json_output without loading the NER model."""

    @staticmethod
    def _parse(reply: str, fallback: str = "ctx"):
        from symptoms_recognizer.ner_model.model import PhenotypesDetector

        return PhenotypesDetector._parse_llm_json_output(None, reply, fallback)

    def test_extracts_mentions_with_context(self):
        reply = '{"fenotipos": [{"fenotipo": "fiebre", "contexto": "tiene fiebre"}]}'
        assert self._parse(reply) == [("fiebre", "tiene fiebre")]

    def test_bare_strings_fall_back_to_chunk_context(self):
        reply = '{"fenotipos": ["fiebre", "convulsiones"]}'
        assert self._parse(reply, "texto") == [("fiebre", "texto"), ("convulsiones", "texto")]

    def test_order_is_stable_and_duplicates_collapse(self):
        # A set() here previously made identical inputs disagree run to run.
        reply = (
            '{"fenotipos": ['
            '{"fenotipo": "fiebre", "contexto": "c1"},'
            '{"fenotipo": "tos", "contexto": "c2"},'
            '{"fenotipo": "fiebre", "contexto": "c1"},'
            '{"fenotipo": "cefalea", "contexto": "c3"}]}'
        )
        expected = [("fiebre", "c1"), ("tos", "c2"), ("cefalea", "c3")]
        assert self._parse(reply) == expected
        assert [self._parse(reply) for _ in range(5)] == [expected] * 5

    def test_empty_phenotype_list_is_a_valid_answer(self):
        assert self._parse('{"fenotipos": []}') == []

    def test_unusable_reply_raises_rather_than_returning_empty(self):
        with pytest.raises(LLMInvalidResponseError):
            self._parse("el modelo no respondio nada util")


class TestProviderContract:
    def test_stub_records_prompts(self):
        provider = StubProvider(replies=['{"hpo_code": "HP:0001250"}'])
        out = provider.complete("sys", "user")
        assert parse_json_object(out)["hpo_code"] == "HP:0001250"
        assert provider.calls == [("sys", "user")]

    def test_provider_errors_propagate(self):
        provider = StubProvider(error=LLMError("provider unreachable"))
        with pytest.raises(LLMError):
            provider.complete("sys", "user")

    def test_name_identifies_provider_and_model(self):
        assert StubProvider().name == "StubProvider(stub)"
