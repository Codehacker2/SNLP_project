"""Optional composition around the original Digitizer, with a separate response."""
from __future__ import annotations

from ..digitizer import Digitizer, expand_pages
from .figures import FigureDetector
from .ner import SpacyNER


class DocumentAnalyzer:
    """Keep schema-1.0 output intact under 'digitization'; add separate annotations.

    Existing callers continue to use Digitizer unchanged. Reuse this object so
    OCR and NER models are loaded once. Components can be supplied independently.
    """

    def __init__(self, digitizer=None, *, ner=None, figures=None):
        self._digitizer = digitizer
        self.ner = ner if ner is not None else SpacyNER()
        self.figures = figures if figures is not None else FigureDetector()

    def analyze(self, pages, *, detect_entities: bool = True, detect_figures: bool = True, **digitize_options) -> dict:
        """Run normal OCR and then optional annotation on those same ordered pages."""
        expanded = expand_pages(pages)
        if not expanded:
            raise ValueError("At least one page is required")
        if detect_entities and isinstance(self.ner, SpacyNER):
            self.ner._load()
        if self._digitizer is None:
            self._digitizer = Digitizer()
        result = self._digitizer.digitize(expanded, **digitize_options)
        return self.annotate(result, expanded, detect_entities=detect_entities, detect_figures=detect_figures)

    def annotate(self, result: dict, pages=None, *, detect_entities: bool = True, detect_figures: bool = True) -> dict:
        """Annotate a previous/reviewed result without rerunning OCR or editing it.

        Supply the original ordered pages for figures. Entity offsets reference
        answer_text exactly; answer_index links back to digitization.answers.
        """
        expanded = expand_pages(pages) if detect_figures and pages is not None else []
        if detect_figures and len(expanded) != result["pages"]:
            raise ValueError("Figure analysis requires the original ordered pages matching result['pages']")
        answers = result["answers"]
        entities = []
        if detect_entities:
            batches = self.ner.extract_many(answer["answer_text"] for answer in answers)
            if len(batches) != len(answers):
                raise ValueError("NER must return one entity list per answer")
            for index, batch in enumerate(batches):
                entities.extend(dict(entity, answer_index=index) for entity in batch)
        figures = []
        if detect_figures:
            for page_number, source in enumerate(expanded, 1):
                figures.extend(self.figures.detect(source, page=page_number))
        return {
            "digitization": result,
            "analysis": {
                "schema_version": "1.0",
                "entities": entities,
                "figures": figures,
                "ner_enabled": detect_entities,
                "figure_detection_enabled": detect_figures,
                "figure_coordinate_space": "deskewed_standard_width_page",
            },
        }
