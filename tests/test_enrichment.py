"""Optional annotation tests: synthetic pages and injected OCR, with no downloads."""
import copy

import cv2
import numpy as np
import pytest

from exam_digitizer.enrichment import DocumentAnalyzer, FigureDetector, NERUnavailableError, SpacyNER
from exam_digitizer.preprocessing.page import Prepped, preprocess
from exam_digitizer.schema import EXAMPLE_RESULT
from tests.conftest import make_page


def prepared(gray):
    """Already deskewed synthetic image, so detector geometry can be tested exactly."""
    return Prepped(gray, gray.copy(), (gray < 185).astype(np.uint8), 0.0, None, [], None)


def blank():
    return np.full((1200, 900), 255, dtype=np.uint8)


def overlaps(box, target):
    x0, y0, x1, y1 = box
    a0, b0, a1, b1 = target
    intersection = max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0))
    return intersection / ((a1 - a0) * (b1 - b0))


@pytest.mark.parametrize("kind", ["box", "circle", "graph", "flowchart"])
def test_detects_common_connected_drawings_and_keeps_pixels_unchanged(kind):
    image = blank()
    if kind == "box":
        cv2.rectangle(image, (250, 350), (600, 650), 0, 4)
    elif kind == "circle":
        cv2.circle(image, (425, 500), 150, 0, 4)
    elif kind == "graph":
        cv2.line(image, (250, 350), (250, 650), 0, 4)
        cv2.line(image, (250, 650), (600, 650), 0, 4)
        cv2.line(image, (250, 620), (560, 380), 0, 4)
    else:
        cv2.rectangle(image, (300, 350), (550, 430), 0, 4)
        cv2.arrowedLine(image, (425, 430), (425, 550), 0, 4)
        cv2.rectangle(image, (300, 550), (550, 650), 0, 4)
    before = image.copy()
    regions = FigureDetector().detect(prepared(image), page=2)
    assert len(regions) == 1
    assert regions[0]["page"] == 2
    assert regions[0]["needs_review"] is True
    assert regions[0]["score_kind"] == "heuristic"
    assert 0 <= regions[0]["score"] <= 1
    assert overlaps(regions[0]["bbox"], (300, 380, 550, 620)) > 0.9
    np.testing.assert_array_equal(image, before)


def test_two_separate_figures_return_two_ordered_boxes():
    image = blank()
    cv2.circle(image, (250, 300), 100, 0, 4)
    cv2.rectangle(image, (500, 700), (750, 950), 0, 4)
    regions = FigureDetector().detect(prepared(image))
    assert len(regions) == 2
    assert regions[0]["bbox"][1] < regions[1]["bbox"][1]


def test_blank_text_and_page_rulings_do_not_become_figures():
    detector = FigureDetector()
    assert detector.detect(prepared(blank())) == []
    image = blank()
    for y in range(200, 1000, 70):
        cv2.line(image, (30, y + 12), (870, y + 12), 200, 1)
        cv2.putText(image, "This is a normal answer line.", (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 1)
    cv2.line(image, (75, 50), (75, 1150), 0, 2)
    assert detector.detect(prepared(image)) == []
    assert detector.detect(preprocess(make_page())) == []


def test_detector_handles_image_bytes_and_standard_page_coordinates():
    image = blank()
    cv2.rectangle(image, (250, 350), (600, 650), 0, 4)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    regions = FigureDetector().detect(encoded.tobytes())
    assert regions
    assert all(0 <= r["bbox"][0] < r["bbox"][2] <= 1654 for r in regions)


def test_detector_does_not_return_header_boxes_or_footer():
    image = blank()
    cv2.rectangle(image, (200, 20), (500, 150), 0, 4)
    pp = prepared(image)
    pp.header_bottom = 180
    assert FigureDetector().detect(pp) == []


def test_ner_preserves_unicode_offsets_and_repeated_mentions():
    spacy = pytest.importorskip("spacy")
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "PERSON", "pattern": "Marie Curie"}, {"label": "GPE", "pattern": "Paris"}])
    reader = SpacyNER(nlp=nlp)
    text = "Résumé: Marie Curie visited Paris. Marie Curie returned."
    entities, empty = reader.extract_many([text, ""])
    assert [e["label"] for e in entities] == ["PERSON", "GPE", "PERSON"]
    assert empty == []
    for entity in entities:
        assert text[entity["start"]:entity["end"]] == entity["text"]
        assert "confidence" not in entity


def test_real_trained_english_ner_model():
    pytest.importorskip("en_core_web_sm")
    entities = SpacyNER().extract("Barack Obama visited Paris in 2010.")
    found = {(entity["text"], entity["label"]) for entity in entities}
    assert ("Barack Obama", "PERSON") in found
    assert ("Paris", "GPE") in found
    assert ("2010", "DATE") in found


def test_missing_ner_model_has_an_actionable_error():
    pytest.importorskip("spacy")
    with pytest.raises(NERUnavailableError, match="python -m spacy download"):
        SpacyNER("missing_exam_ner_model").extract("A sample answer")


class FakeNER:
    def extract_many(self, texts):
        return [[{"text": text[:3], "label": "TEST", "start": 0, "end": 3}] if text else [] for text in texts]


class FailingComponent:
    def _load(self):
        raise AssertionError("Disabled component was loaded")

    def extract_many(self, texts):
        raise AssertionError("Disabled NER was called")

    def detect(self, source, page):
        raise AssertionError("Disabled figure detector was called")


def test_annotation_keeps_original_result_and_review_decisions_unchanged():
    result = copy.deepcopy(EXAMPLE_RESULT)
    before = copy.deepcopy(result)
    analyzer = DocumentAnalyzer(ner=FakeNER())
    enriched = analyzer.annotate(result, [prepared(blank())])
    assert enriched["digitization"] is result
    assert result == before
    assert len(enriched["analysis"]["entities"]) == 2
    assert [e["answer_index"] for e in enriched["analysis"]["entities"]] == [0, 1]
    assert enriched["analysis"]["figures"] == []


def test_optional_features_can_be_disabled_without_loading_them():
    analyzer = DocumentAnalyzer(ner=FailingComponent(), figures=FailingComponent())
    enriched = analyzer.annotate(copy.deepcopy(EXAMPLE_RESULT), detect_entities=False, detect_figures=False)
    assert enriched["analysis"]["entities"] == []
    assert enriched["analysis"]["figures"] == []
    assert not enriched["analysis"]["ner_enabled"]


def test_figure_annotation_requires_the_matching_page_count():
    with pytest.raises(ValueError, match="original ordered pages"):
        DocumentAnalyzer().annotate(EXAMPLE_RESULT, [], detect_entities=False)


def test_analyzer_calls_existing_digitizer_once_and_accepts_pdf(tmp_path):
    from PIL import Image

    class ExistingDigitizer:
        def __init__(self):
            self.calls = []

        def digitize(self, pages, **kwargs):
            self.calls.append((pages, kwargs))
            return copy.deepcopy(EXAMPLE_RESULT)

    path = tmp_path / "input.pdf"
    Image.fromarray(blank()).save(path, "PDF", resolution=200)
    digitizer = ExistingDigitizer()
    result = DocumentAnalyzer(digitizer, ner=FakeNER()).analyze(path, booklet_id="DEMO", detect_figures=False)
    assert len(digitizer.calls) == 1
    pages, options = digitizer.calls[0]
    assert len(pages) == 1 and isinstance(pages[0], np.ndarray)
    assert options == {"booklet_id": "DEMO"}
    assert result["digitization"] == EXAMPLE_RESULT


def test_empty_page_list_is_rejected_before_ocr():
    with pytest.raises(ValueError, match="At least one page"):
        DocumentAnalyzer().analyze([], detect_entities=False)
