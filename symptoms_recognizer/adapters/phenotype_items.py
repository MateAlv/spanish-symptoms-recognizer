"""Turn recognizer output into the /extract-hpo response shape.

The recognizer yields `(mention, hpo_code)` pairs. The gateway's
`PhenotypeItem` has thirteen fields, so the gap is filled here rather than
inside the recognizer, which keeps the module independent of one consumer's
contract.
"""
from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass

from symptoms_recognizer.adapters.hpo_labels import HpoLabelIndex, default_label_index

NO_OFFSET = -1

# How the HPO code was arrived at. The incumbent reports its matcher here
# ("exact", "fuzzy"); this pipeline always reaches the code the same way.
MATCHED_BY_LLM_RAG = "llm_rag"


@dataclass(frozen=True)
class PhenotypeItem:
    """Mirrors services/api-gateway/app/schemas.py:PhenotypeItem."""

    hpo_id: str
    label: str
    label_es: str
    span_es: str
    span_en: str
    start_es: int
    end_es: int
    start_en: int
    end_en: int
    negated: bool
    confidence: float
    matched_by: str

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(text: str) -> str:
    """Casefold and strip accents, so 'Hipotonía' can match 'hipotonia'."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def find_span(text: str, mention: str) -> tuple[int, int]:
    """Locate `mention` in `text`, returning (-1, -1) when it is not there.

    The NER stage normalises what it reports -- "Sensación febril" comes back
    as "fiebre" -- so a literal search misses often enough that an accent- and
    case-insensitive pass is worth the cost. A mention the model rephrased is
    reported without offsets rather than with guessed ones: the incumbent
    returns -1 for every offset today, so this is no worse, and a wrong
    highlight is worse than none.
    """
    if not text or not mention:
        return NO_OFFSET, NO_OFFSET

    index = text.find(mention)
    if index != -1:
        return index, index + len(mention)

    # Accent-insensitive fallback. Normalisation is per character here, so
    # offsets still refer to the original string.
    haystack = _normalize(text)
    needle = _normalize(mention)
    if not needle:
        return NO_OFFSET, NO_OFFSET

    index = haystack.find(needle)
    if index == -1:
        return NO_OFFSET, NO_OFFSET

    return index, index + len(needle)


def build_phenotype_items(
    scan_results: list[tuple[str, str]],
    source_text: str = "",
    label_index: HpoLabelIndex | None = None,
    confidence: float = 1.0,
) -> list[PhenotypeItem]:
    """Build response items from `PhenotypesRecognizer.scan()` output.

    `negated` is always False: the NER stage is instructed to drop negated
    findings, so a mention that reaches here is asserted, not denied. That
    matches what the incumbent returns, though for a different reason -- its
    negation detector is not wired into the live path at all.
    """
    labels = label_index or default_label_index()

    items = []
    for mention, hpo_id in scan_results:
        start, end = find_span(source_text, mention)
        items.append(
            PhenotypeItem(
                hpo_id=hpo_id,
                label=labels.english(hpo_id),
                label_es=labels.spanish(hpo_id),
                span_es=mention,
                span_en="",  # nothing is translated on this path
                start_es=start,
                end_es=end,
                start_en=NO_OFFSET,
                end_en=NO_OFFSET,
                negated=False,
                confidence=confidence,
                matched_by=MATCHED_BY_LLM_RAG,
            )
        )
    return items
