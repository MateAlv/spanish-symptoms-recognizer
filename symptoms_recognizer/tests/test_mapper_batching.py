"""Offline tests for batched HPO code resolution.

Batching trades calls for the risk that one reply is misread across several
mentions, so the alignment and partial-answer paths are what these cover.
None of them load the encoder or reach a provider.
"""
from __future__ import annotations

import json

import pytest

from symptoms_recognizer.llm.base import LLMError, LLMProvider
import symptoms_recognizer.mapper.mapper as mapper_module
from symptoms_recognizer.mapper.mapper import PhenotypeOntologyMapper


class ScriptedProvider(LLMProvider):
    def __init__(self, replies=None, error=None):
        super().__init__(model="scripted")
        self.replies = list(replies or [])
        self.error = error
        self.prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        if self.error:
            raise self.error
        return self.replies.pop(0) if self.replies else '{"codigos": {}}'

    @property
    def calls(self) -> int:
        return len(self.prompts)


def make_mapper(provider, prompt=None):
    """A mapper with the encoder and ontology bypassed."""
    mapper = PhenotypeOntologyMapper.__new__(PhenotypeOntologyMapper)
    mapper.api_provider = "stub"
    mapper.api_model_name = "stub"
    mapper.provider = provider
    mapper.prompt = prompt
    mapper.top_k = 3
    mapper.last_mapping_logs = []
    return mapper


@pytest.fixture
def batching(monkeypatch):
    """Turn batching on for the tests that exercise it.

    The shipped default is one call per mention -- see MAP_BATCH_SIZE -- so
    these tests set it explicitly rather than depending on it.
    """
    monkeypatch.setattr(mapper_module, "MAP_BATCH_SIZE", 12)
    return 12


def group(*names):
    """Build `to_resolve` entries: (index, mention, context, candidates)."""
    return [
        (i, name, f"contexto de {name}", f"1. Código: HP:000000{i} | Nombre: {name}\n")
        for i, name in enumerate(names)
    ]


def codes_reply(mapping: dict) -> str:
    return json.dumps({"codigos": {str(k): v for k, v in mapping.items()}})


class TestBatching:
    def test_one_call_resolves_every_mention(self, batching):
        provider = ScriptedProvider([codes_reply({0: "HP:0001945", 1: "HP:0000252"})])
        mapper = make_mapper(provider)

        codes = ["None", "None"]
        logs = [None, None]
        mapper._resolve_codes(group("fiebre", "microcefalia"), codes, logs)

        assert provider.calls == 1
        assert codes == ["HP:0001945", "HP:0000252"]

    def test_all_mentions_appear_in_the_single_prompt(self, batching):
        provider = ScriptedProvider([codes_reply({0: "HP:1", 1: "HP:2"})])
        mapper = make_mapper(provider)
        mapper._resolve_codes(group("fiebre", "microcefalia"), ["None"] * 2, [None] * 2)

        prompt = provider.prompts[0]
        assert "fiebre" in prompt and "microcefalia" in prompt
        assert "### Fenotipo 0" in prompt and "### Fenotipo 1" in prompt

    def test_groups_are_split_at_the_batch_size(self, batching):
        count = mapper_module.MAP_BATCH_SIZE + 3
        provider = ScriptedProvider([
            codes_reply({i: f"HP:{i:07d}" for i in range(mapper_module.MAP_BATCH_SIZE)}),
            codes_reply({i: f"HP:{i:07d}" for i in range(3)}),
        ])
        mapper = make_mapper(provider)

        codes = ["None"] * count
        mapper._resolve_codes(group(*[f"m{i}" for i in range(count)]), codes, [None] * count)

        assert provider.calls == 2
        assert "None" not in codes

    def test_indices_restart_within_each_group(self, batching):
        # Keys are positions inside a group, so the second call's keys start
        # at 0 again -- a global index would silently drop every later code.
        count = mapper_module.MAP_BATCH_SIZE + 2
        provider = ScriptedProvider([
            codes_reply({i: "HP:0000001" for i in range(mapper_module.MAP_BATCH_SIZE)}),
            codes_reply({0: "HP:0000042", 1: "HP:0000043"}),
        ])
        mapper = make_mapper(provider)

        codes = ["None"] * count
        mapper._resolve_codes(group(*[f"m{i}" for i in range(count)]), codes, [None] * count)

        assert codes[mapper_module.MAP_BATCH_SIZE] == "HP:0000042"
        assert codes[mapper_module.MAP_BATCH_SIZE + 1] == "HP:0000043"


class TestPartialAndMalformedReplies:
    def test_missing_entry_becomes_none_without_shifting_others(self, batching):
        # The model answers 0 and 2 but not 1. Mention 2 must keep its own
        # code rather than inheriting the answer meant for 1.
        provider = ScriptedProvider([codes_reply({0: "HP:0000001", 2: "HP:0000003"})])
        mapper = make_mapper(provider)

        codes = ["None"] * 3
        mapper._resolve_codes(group("a", "b", "c"), codes, [None] * 3)

        assert codes == ["HP:0000001", "None", "HP:0000003"]

    def test_unparseable_reply_yields_no_codes_rather_than_wrong_ones(self, batching):
        provider = ScriptedProvider(["el modelo divagó y no devolvió JSON"])
        mapper = make_mapper(provider)

        codes = ["None"] * 2
        mapper._resolve_codes(group("a", "b"), codes, [None] * 2)

        assert codes == ["None", "None"]

    @pytest.mark.parametrize(
        "reply",
        [
            '{"codigos": ["HP:0000001"]}',       # list, not an object
            '{"otra_clave": {"0": "HP:1"}}',     # wrong key
            '{"codigos": {"no_es_un_indice": "HP:1"}}',
            '{"codigos": {"99": "HP:1"}}',       # out of range
        ],
    )
    def test_malformed_shapes_do_not_assign_codes(self, reply, batching):
        mapper = make_mapper(ScriptedProvider([reply]))
        codes = ["None"]
        mapper._resolve_codes(group("a"), codes, [None])
        assert codes == ["None"]

    def test_non_string_code_is_ignored(self, batching):
        mapper = make_mapper(ScriptedProvider(['{"codigos": {"0": 12345}}']))
        codes = ["None"]
        mapper._resolve_codes(group("a"), codes, [None])
        assert codes == ["None"]

    def test_provider_failure_propagates(self, batching):
        mapper = make_mapper(ScriptedProvider(error=LLMError("unreachable")))
        with pytest.raises(LLMError):
            mapper._resolve_codes(group("a"), ["None"], [None])


class TestShippedDefault:
    """MAP_BATCH_SIZE is 1, so the default path is one call per mention.

    Batching measured slower on TARS -- short concurrent calls beat one long
    one -- so these cover what actually ships.
    """

    def test_default_is_one_call_per_mention(self):
        assert mapper_module.MAP_BATCH_SIZE == 1

        provider = ScriptedProvider(['{"hpo_code": "HP:0001945"}'] * 3)
        mapper = make_mapper(provider)

        codes = ["None"] * 3
        mapper._resolve_codes(group("a", "b", "c"), codes, [None] * 3)

        assert provider.calls == 3
        assert codes == ["HP:0001945"] * 3

    def test_each_prompt_carries_only_its_own_mention(self):
        provider = ScriptedProvider(['{"hpo_code": "HP:0001945"}'] * 2)
        mapper = make_mapper(provider)
        mapper._resolve_codes(group("fiebre", "microcefalia"), ["None"] * 2, [None] * 2)

        for prompt in provider.prompts:
            assert "### Fenotipo" not in prompt  # not the batched shape
        joined = " ".join(provider.prompts)
        assert "fiebre" in joined and "microcefalia" in joined

    def test_none_is_recorded_when_no_candidate_fits(self):
        provider = ScriptedProvider(['{"hpo_code": "None"}'])
        mapper = make_mapper(provider)
        codes, logs = ["None"], [None]
        mapper._resolve_codes(group("algo raro"), codes, logs)
        assert codes == ["None"]
        assert logs[0]["final_code"] == "None"


class TestLogsAndCustomPrompt:
    def test_logs_record_each_mention(self, batching):
        provider = ScriptedProvider([codes_reply({0: "HP:0001945", 1: "None"})])
        mapper = make_mapper(provider)

        logs = [None, None]
        mapper._resolve_codes(group("fiebre", "raro"), ["None"] * 2, logs)

        assert [log["phenotype"] for log in logs] == ["fiebre", "raro"]
        assert [log["final_code"] for log in logs] == ["HP:0001945", "None"]

    def test_custom_prompt_still_calls_once_per_mention(self):
        # A caller's own prompt describes a single mention, so batching it
        # would change what that prompt means.
        provider = ScriptedProvider(['{"hpo_code": "HP:0000001"}'] * 3)
        mapper = make_mapper(provider, prompt="mi prompt para UN fenotipo")

        codes = ["None"] * 3
        mapper._resolve_codes(group("a", "b", "c"), codes, [None] * 3)

        assert provider.calls == 3
        assert codes == ["HP:0000001"] * 3
