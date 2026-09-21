"""English and Spanish HPO labels, read from the Babelon translation file."""
from __future__ import annotations

import csv
import os
from functools import lru_cache

MAPPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mapper")
DEFAULT_TSV_PATH = os.path.join(MAPPER_DIR, "hpo/hp-es.babelon.tsv")

HPO_CODE_COLUMN = "subject_id"
ENGLISH_LABEL_COLUMN = "source_value"
SPANISH_LABEL_COLUMN = "translation_value"


class HpoLabelIndex:
    """Looks up the English and Spanish label for an HPO code.

    The embedding batches carry only the Spanish label, and the response
    contract asks for both, so the labels are read from the same Babelon file
    the embeddings were built from. That keeps them consistent by construction.
    """

    def __init__(self, tsv_path: str | None = None):
        self.tsv_path = tsv_path or DEFAULT_TSV_PATH
        self._labels: dict[str, tuple[str, str]] = {}

        if not os.path.exists(self.tsv_path):
            raise FileNotFoundError(
                f"HPO label file not found at {self.tsv_path}. "
                "Download hp-es.babelon.tsv from obophenotype/hpo-translations."
            )

        with open(self.tsv_path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                code = row[HPO_CODE_COLUMN].strip()
                if code and code not in self._labels:
                    self._labels[code] = (
                        row[ENGLISH_LABEL_COLUMN].strip(),
                        row[SPANISH_LABEL_COLUMN].strip(),
                    )

    def english(self, hpo_id: str) -> str:
        return self._labels.get(hpo_id, ("", ""))[0]

    def spanish(self, hpo_id: str) -> str:
        return self._labels.get(hpo_id, ("", ""))[1]

    def __len__(self) -> int:
        return len(self._labels)


@lru_cache(maxsize=1)
def default_label_index() -> HpoLabelIndex:
    """The shared index. Parsing ~20k rows per request would be wasteful."""
    return HpoLabelIndex()
