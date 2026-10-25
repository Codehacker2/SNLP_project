"""Local named-entity recognition on OCR text, with answer-relative offsets."""
from __future__ import annotations

from typing import Iterable


class NERUnavailableError(RuntimeError):
    """The optional NER dependency or trained model is unavailable."""


class SpacyNER:
    """Recognize people, organizations, places, dates, etc. using a trained pipeline.

    Loading is lazy. No package/model is downloaded automatically. An injected
    spaCy Language pipeline is supported for testing and custom trained models.
    spaCy does not provide calibrated per-entity probabilities; none are invented.
    """

    def __init__(self, model: str = "en_core_web_sm", *, nlp=None, batch_size: int = 16):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model
        self._nlp = nlp
        self.batch_size = batch_size

    def _load(self):
        if self._nlp is None:
            try:
                import spacy
            except ImportError as exc:
                raise NERUnavailableError(
                    'NER requires the optional dependency: pip install -e ".[ner]"'
                ) from exc
            try:
                nlp = spacy.load(self.model)
            except OSError as exc:
                raise NERUnavailableError(
                    f"NER model {self.model!r} is not installed. For the default model, "
                    "run: python -m spacy download en_core_web_sm. "
                    "A local trained model directory can also be supplied."
                ) from exc
            if "ner" not in nlp.pipe_names:
                raise NERUnavailableError(f"Model {self.model!r} has no trained 'ner' component")
            self._nlp = nlp
        return self._nlp

    def extract_many(self, texts: Iterable[str]) -> list[list[dict]]:
        """Offsets refer to the exact input string, without text normalization."""
        texts = list(texts)
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("NER inputs must be strings")
        nlp = self._load()
        return [
            [
                {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                for ent in doc.ents
            ]
            for doc in nlp.pipe(texts, batch_size=self.batch_size)
        ]

    def extract(self, text: str) -> list[dict]:
        return self.extract_many([text])[0]
