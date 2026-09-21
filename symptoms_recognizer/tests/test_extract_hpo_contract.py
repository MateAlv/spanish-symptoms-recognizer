"""Contract tests for the /extract-hpo response.

Where the gateway's schemas.py can be found, the response is validated
against the real pydantic models rather than against a local restatement of
them. A hand-copied field list drifts; these fail when the contract moves.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from symptoms_recognizer.adapters.extract_hpo import RecognizerExtractor
from symptoms_recognizer.adapters.hpo_labels import HpoLabelIndex

GATEWAY_SCHEMAS = os.path.expanduser(
    "~/diagnostic.ar/diagnostic.ar-api/services/api-gateway/app/schemas.py"
)


def load_gateway_schemas():
    """Import the gateway's schemas.py directly, or skip."""
    if not os.path.exists(GATEWAY_SCHEMAS):
        return None
    try:
        spec = importlib.util.spec_from_file_location("gateway_schemas", GATEWAY_SCHEMAS)
        module = importlib.util.module_from_spec(spec)
        sys.modules["gateway_schemas"] = module
        spec.loader.exec_module(module)
        return module
    except Exception:  # pydantic missing, or the file moved on
        return None


class StubRecognizer:
    def __init__(self, results):
        self.results = results
        self.seen = []

    def scan(self, text):
        self.seen.append(text)
        return self.results


@pytest.fixture
def labels(tmp_path):
    tsv = tmp_path / "hp-es.babelon.tsv"
    tsv.write_text(
        "subject_id\tsource_value\ttranslation_value\n"
        "HP:0001945\tFever\tFiebre\n"
        "HP:0000252\tMicrocephaly\tMicrocefalia\n",
        encoding="utf-8",
    )
    return HpoLabelIndex(str(tsv))


@pytest.fixture
def extractor(labels):
    return RecognizerExtractor(
        recognizer=StubRecognizer([("fiebre", "HP:0001945"), ("microcefalia", "HP:0000252")]),
        label_index=labels,
        model="claude-opus-5",
    )


class TestResponseShape:
    def test_top_level_keys(self, extractor):
        response = extractor.extract("Presenta fiebre y microcefalia")
        assert set(response) == {"symptoms_extracted", "model", "phenotypes"}

    def test_symptoms_extracted_mirrors_the_spans(self, extractor):
        response = extractor.extract("Presenta fiebre y microcefalia")
        assert response["symptoms_extracted"] == ["fiebre", "microcefalia"]

    def test_model_info_names_each_stage(self, extractor):
        model = extractor.extract("Presenta fiebre")["model"]
        assert set(model) == {"symptom_extractor", "translator", "phenotyper"}
        # This pipeline never translates; saying so beats a plausible lie.
        assert model["translator"] == "none"

    def test_empty_text_returns_an_empty_result_without_calling_the_model(self, labels):
        recognizer = StubRecognizer([])
        extractor = RecognizerExtractor(recognizer=recognizer, label_index=labels)
        response = extractor.extract("   ")
        assert response["phenotypes"] == []
        assert recognizer.seen == []

    def test_offsets_resolve_against_offset_text_when_given(self, labels):
        # The gateway extracts from a normalised copy, which regex rewriting
        # can lengthen or shorten, and needs offsets into the user's own text.
        extractor = RecognizerExtractor(
            recognizer=StubRecognizer([("fiebre", "HP:0001945")]), label_index=labels
        )
        original = "Paciente  con  fiebre"
        response = extractor.extract("Paciente con fiebre", offset_text=original)

        item = response["phenotypes"][0]
        assert original[item["start_es"]:item["end_es"]] == "fiebre"

    def test_offsets_fall_back_to_the_extracted_text(self, labels):
        extractor = RecognizerExtractor(
            recognizer=StubRecognizer([("fiebre", "HP:0001945")]), label_index=labels
        )
        text = "Presenta fiebre"
        item = extractor.extract(text)["phenotypes"][0]
        assert text[item["start_es"]:item["end_es"]] == "fiebre"

    def test_provider_failure_propagates(self, labels):
        class Failing:
            def scan(self, text):
                raise RuntimeError("provider unreachable")

        extractor = RecognizerExtractor(recognizer=Failing(), label_index=labels)
        with pytest.raises(RuntimeError):
            extractor.extract("Presenta fiebre")


class TestAgainstGatewaySchema:
    """Validates against the gateway's own pydantic models when available."""

    @pytest.fixture(autouse=True)
    def schemas(self):
        module = load_gateway_schemas()
        if module is None:
            pytest.skip("gateway schemas.py not available")
        return module

    def test_response_validates_as_extract_response(self, extractor, schemas):
        response = extractor.extract("Presenta fiebre y microcefalia")
        parsed = schemas.ExtractResponse(**response)
        assert len(parsed.phenotypes) == 2

    def test_every_phenotype_item_validates(self, extractor, schemas):
        response = extractor.extract("Presenta fiebre y microcefalia")
        for item in response["phenotypes"]:
            schemas.PhenotypeItem(**item)

    def test_no_extra_or_missing_phenotype_fields(self, extractor, schemas):
        response = extractor.extract("Presenta fiebre")
        expected = set(schemas.PhenotypeItem.model_fields)
        assert set(response["phenotypes"][0]) == expected

    def test_empty_result_still_validates(self, labels, schemas):
        extractor = RecognizerExtractor(
            recognizer=StubRecognizer([]), label_index=labels
        )
        schemas.ExtractResponse(**extractor.extract(""))
