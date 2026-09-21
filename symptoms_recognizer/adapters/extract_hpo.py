"""A drop-in `/extract-hpo` backend built on the recognizer.

Exposes one method returning the gateway's `ExtractResponse` shape, so the
gateway changes at a single seam. Nothing here imports the gateway: this
module is the recognizer's side of the contract, and the wiring lives there
behind a feature flag.
"""
from __future__ import annotations

import logging
import time

from symptoms_recognizer.adapters.hpo_labels import HpoLabelIndex
from symptoms_recognizer.adapters.phenotype_items import (
    PhenotypeItem,
    build_phenotype_items,
)
from symptoms_recognizer.symptom_recognizer import PhenotypesRecognizer

logger = logging.getLogger(__name__)

DEFAULT_TEXT_PARSER = "sentences"


class RecognizerExtractor:
    """Runs the recognizer and returns `/extract-hpo`-shaped results.

    The recognizer is built once and reused: it loads a 485MB encoder and the
    HPO embedding batches, which must not happen per request.
    """

    def __init__(
        self,
        recognizer: PhenotypesRecognizer | None = None,
        label_index: HpoLabelIndex | None = None,
        provider: str | None = None,
        model: str | None = None,
        text_parser: str = DEFAULT_TEXT_PARSER,
    ):
        self.recognizer = recognizer or PhenotypesRecognizer(
            ontology="hpo",
            phenotypes_model_type="api",
            ner_api_provider=provider,
            ner_api_model_name=model,
            map_api_provider=provider,
            map_api_model_name=model,
            text_parser=text_parser,
        )
        self.label_index = label_index
        self.model_name = model or "default"

    def extract(self, text: str, offset_text: str | None = None) -> dict:
        """Extract phenotypes from Spanish clinical text.

        `offset_text` is what character offsets are resolved against, for a
        caller that extracts from a normalised copy: normalisation rewrites
        the text by regex and can change its length, so an offset taken from
        the copy would not line up with what the user actually sent.

        Raises `LLMError` when the model cannot be reached. The caller decides
        what to do about that -- returning an empty phenotype list would read
        as "this patient has no symptoms", which is not a safe substitute for
        an error in a clinical tool.
        """
        if not text or not text.strip():
            return self._response([])

        started = time.monotonic()
        scan_results = self.recognizer.scan(text)
        items = build_phenotype_items(
            scan_results,
            source_text=offset_text if offset_text is not None else text,
            label_index=self.label_index,
        )

        logger.info(
            "extract-hpo phenotypes=%d chars=%d elapsed_ms=%d",
            len(items),
            len(text),
            int((time.monotonic() - started) * 1000),
        )
        return self._response(items)

    def _response(self, items: list[PhenotypeItem]) -> dict:
        # ModelInfo names the three stages the incumbent runs. This pipeline
        # has no separate translator -- it works in Spanish throughout -- and
        # one model does both remaining stages, so that is what is reported
        # rather than inventing names the gateway would not recognise.
        return {
            "symptoms_extracted": [item.span_es for item in items],
            "model": {
                "symptom_extractor": f"recognizer-ner:{self.model_name}",
                "translator": "none",
                "phenotyper": f"clinlinker-kb-gp+{self.model_name}",
            },
            "phenotypes": [item.to_dict() for item in items],
        }
