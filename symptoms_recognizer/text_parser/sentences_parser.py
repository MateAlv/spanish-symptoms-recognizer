from symptoms_recognizer.concurrency import map_in_order
from symptoms_recognizer.ner_model.model import PhenotypesDetector
from symptoms_recognizer.text_parser.text_parser_interface import HistoryRecordParser

import spacy

class SentencesParser(HistoryRecordParser):
    def __init__(self):
        self.text_nlp = spacy.blank("es")
        self.text_nlp.add_pipe("sentencizer")

    def apply(self, text: str, ner_model: PhenotypesDetector):
        doc = self.text_nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # Each sentence is read on its own, so all of them at once, joined in order.
        phenotypes_list = []
        for sentence_results in map_in_order(ner_model.detect_phenotypes, sentences):
            phenotypes_list.extend(sentence_results)

        return phenotypes_list