"""Offline tests for the /extract-hpo response adapter."""
from __future__ import annotations

import pytest

from symptoms_recognizer.adapters.hpo_labels import HpoLabelIndex
from symptoms_recognizer.adapters.phenotype_items import (
    NO_OFFSET,
    build_phenotype_items,
    find_span,
)

GATEWAY_FIELDS = {
    "hpo_id", "label", "label_es", "span_es", "span_en",
    "start_es", "end_es", "start_en", "end_en",
    "negated", "confidence", "matched_by",
}


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


class TestFindSpan:
    def test_exact_match(self):
        assert find_span("Presenta fiebre alta", "fiebre") == (9, 15)

    def test_absent_mention_has_no_offsets(self):
        assert find_span("Presenta fiebre", "convulsiones") == (NO_OFFSET, NO_OFFSET)

    def test_accent_insensitive_match(self):
        # The NER stage often reports a mention without the accent.
        start, end = find_span("Presenta hipotonía generalizada", "hipotonia")
        assert (start, end) == (9, 18)

    def test_offsets_index_the_original_text(self):
        text = "Niño con microcefalia y talla baja"
        start, end = find_span(text, "microcefalia")
        assert text[start:end] == "microcefalia"

    def test_accent_offsets_index_the_original_text(self):
        text = "Se constata hipotonía marcada"
        start, end = find_span(text, "hipotonia")
        # The slice must still line up against the accented original.
        assert text[start:end] == "hipotonía"

    @pytest.mark.parametrize("text,mention", [("", "fiebre"), ("Presenta fiebre", "")])
    def test_empty_inputs(self, text, mention):
        assert find_span(text, mention) == (NO_OFFSET, NO_OFFSET)


class TestBuildPhenotypeItems:
    def test_shape_matches_the_gateway_contract(self, labels):
        items = build_phenotype_items([("fiebre", "HP:0001945")], "Presenta fiebre", labels)
        assert set(items[0].to_dict()) == GATEWAY_FIELDS

    def test_labels_are_populated_from_the_ontology(self, labels):
        item = build_phenotype_items([("fiebre", "HP:0001945")], "Presenta fiebre", labels)[0]
        assert item.label == "Fever"
        assert item.label_es == "Fiebre"

    def test_offsets_are_filled_when_the_mention_is_literal(self, labels):
        item = build_phenotype_items([("fiebre", "HP:0001945")], "Presenta fiebre", labels)[0]
        assert (item.start_es, item.end_es) == (9, 15)

    def test_rephrased_mention_reports_no_offsets(self, labels):
        # Better an absent highlight than one pointing at the wrong words.
        item = build_phenotype_items(
            [("fiebre", "HP:0001945")], "Refiere sensacion febril", labels
        )[0]
        assert (item.start_es, item.end_es) == (NO_OFFSET, NO_OFFSET)

    def test_unknown_code_yields_empty_labels_not_an_error(self, labels):
        item = build_phenotype_items([("algo", "HP:9999999")], "algo", labels)[0]
        assert item.label == "" and item.label_es == ""

    def test_english_offsets_are_absent_on_this_path(self, labels):
        item = build_phenotype_items([("fiebre", "HP:0001945")], "Presenta fiebre", labels)[0]
        assert item.span_en == ""
        assert (item.start_en, item.end_en) == (NO_OFFSET, NO_OFFSET)

    def test_mentions_are_not_marked_negated(self, labels):
        # The NER stage drops negated findings, so anything here is asserted.
        item = build_phenotype_items([("fiebre", "HP:0001945")], "Presenta fiebre", labels)[0]
        assert item.negated is False

    def test_order_and_count_are_preserved(self, labels):
        results = [("fiebre", "HP:0001945"), ("microcefalia", "HP:0000252")]
        items = build_phenotype_items(results, "fiebre y microcefalia", labels)
        assert [i.hpo_id for i in items] == ["HP:0001945", "HP:0000252"]

    def test_empty_input(self, labels):
        assert build_phenotype_items([], "", labels) == []
