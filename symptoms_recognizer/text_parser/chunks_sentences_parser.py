from symptoms_recognizer.concurrency import map_in_order
from symptoms_recognizer.ner_model.model import PhenotypesDetector
from symptoms_recognizer.text_parser.text_parser_interface import HistoryRecordParser
import spacy

class ChunkSentencesParser(HistoryRecordParser):
    def __init__(self, max_chunk_tokens: int = 384, overlap_sentences: int = 1):
        self.text_nlp = spacy.blank("es")
        self.text_nlp.add_pipe("sentencizer")
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_sentences = overlap_sentences

    def _build_chunks(self, sentences: list[str], tokenizer) -> list[str]:
        chunks = []
        current_chunk_sents = []
        current_tokens_count = 0

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_token_len = len(tokenizer.encode(sent, add_special_tokens=False))

            if sent_token_len > self.max_chunk_tokens:
                if current_chunk_sents:
                    chunks.append(" ".join(current_chunk_sents))
                    current_chunk_sents = []
                    current_tokens_count = 0
                chunks.append(sent)
                i += 1
                continue

            if current_tokens_count + sent_token_len <= self.max_chunk_tokens:
                current_chunk_sents.append(sent)
                current_tokens_count += sent_token_len
                i += 1
            else:
                chunks.append(" ".join(current_chunk_sents))

                step_back = min(self.overlap_sentences, len(current_chunk_sents))
                if step_back > 0:
                    current_chunk_sents = current_chunk_sents[-step_back:]
                    current_tokens_count = sum(
                        len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk_sents
                    )
                else:
                    current_chunk_sents = []
                    current_tokens_count = 0

        if current_chunk_sents:
            chunks.append(" ".join(current_chunk_sents))

        return chunks

    def apply(self, text: str, ner_model: PhenotypesDetector) -> list[str]:
        doc = self.text_nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return []

        chunks = self._build_chunks(sentences, ner_model.tokenizer)

        # Each chunk is read on its own, so all of them at once, joined in order.
        raw_phenotypes = []
        for chunk_results in map_in_order(ner_model.detect_phenotypes, chunks):
            raw_phenotypes.extend(chunk_results)

        seen = set()
        unique_phenotypes = []
        for p in raw_phenotypes:
            normalized = p.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_phenotypes.append(p.strip())

        return unique_phenotypes