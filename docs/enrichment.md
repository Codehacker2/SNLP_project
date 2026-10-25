# Optional NER and figure detection

These features are an additional layer around the existing application. The original
`Digitizer`, its schema-1.0 answers, its review decisions, and its HTTP routes are
unchanged. Data, sample booklets, results and OCR model assets are unchanged.

## Install

```bash
python -m pip install -e ".[api,pdf,ner]"
python -m spacy download en_core_web_sm
```

The second command installs the trained English NER model. It is a one-time download;
runtime analysis stays local and never installs or downloads anything automatically.
If a model is missing, NER returns an actionable error instead of an empty result.
Figure detection uses the existing OpenCV dependency and needs no extra model.
The optional extra pins a compatible `preshed` release on Windows; other platforms
use spaCy's dependency resolution.

## Python integration

```python
from exam_digitizer import Digitizer
from exam_digitizer.enrichment import DocumentAnalyzer

digitizer = Digitizer()  # Or pass the Digitizer your application already reuses.
analyzer = DocumentAnalyzer(digitizer)
result = analyzer.analyze(
    ["page1.png", "page2.png"],
    booklet_id="T001",
    roster=["23D001", "23D019"],
)

original_output = result["digitization"]
entities = result["analysis"]["entities"]
figure_regions = result["analysis"]["figures"]
```

Either feature can be disabled independently:

```python
result = analyzer.analyze("booklet.pdf", detect_entities=False)
result = analyzer.analyze("booklet.pdf", detect_figures=False)
```

To annotate an existing or human-corrected result without running OCR again:

```python
result = analyzer.annotate(reviewed_result, ["page1.png", "page2.png"])

# Text-only NER needs no page images:
result = analyzer.annotate(reviewed_result, detect_figures=False)
```

Original pages must be supplied in the same order, with the same total page count.
The annotation layer does not modify the supplied result.

## Output

The result is an envelope with two separate objects:

```json
{
  "digitization": {"schema_version": "1.0", "answers": ["...original answer records..."]},
  "analysis": {
    "schema_version": "1.0",
    "entities": [
      {"answer_index": 0, "text": "Marie Curie", "label": "PERSON", "start": 0, "end": 11}
    ],
    "figures": [
      {"page": 1, "bbox": [250, 350, 605, 655], "label": "figure",
       "score": 0.8, "score_kind": "heuristic", "needs_review": true}
    ],
    "ner_enabled": true,
    "figure_detection_enabled": true,
    "figure_coordinate_space": "deskewed_standard_width_page"
  }
}
```

This is an abbreviated illustration; `digitization` contains the complete original
result. Send `result["digitization"]` to existing grading/review consumers.

Entity `answer_index` is zero-based and refers to `digitization.answers`. Character
offsets are zero-based, end-exclusive positions in that answer's exact `answer_text`.
The entity text is not normalized. An entity's page is not guessed when an answer
continues across pages. The trained NER model predicts people, organizations, places,
dates and other English entity types; accuracy depends on OCR quality. Per-entity
confidence is omitted because the spaCy interface does not provide that probability.

Figure boxes are `[x0, y0, x1, y1]`, end-exclusive, on the preprocessed, cropped and
deskewed page with standard width 1654 pixels, matching the existing `/preview`
coordinate system. Multiply coordinates by the preview scale to draw overlays.

## HTTP integration

Start the optional app instead of the original app when you want the new routes:

```bash
python -m uvicorn exam_digitizer.enrichment.api:app --port 8000
```

All original routes, including `/digitize`, `/digitize-base64`, `/ui` and `/preview`,
remain available. The original review UI retains its current behavior. New analysis
is exposed through `/analyze`, `/analyze-base64` and the interactive `/docs` page.

`POST /analyze` accepts the same multipart fields as `/digitize`, plus boolean
`detect_entities` and `detect_figures` fields, both defaulting to true. Example:

```bash
curl -F "pages=@page1.png" -F "booklet_id=T001" -F "detect_entities=true" \
     -F "detect_figures=true" http://127.0.0.1:8000/analyze
```

`POST /analyze-base64` accepts the same JSON as `/digitize-base64`, plus the two
boolean options. Existing file-size and page-count limits also apply. An unavailable
NER dependency/model returns HTTP 503; set `detect_entities=false` for figure-only
analysis. Figure scores are ranking values, not calibrated probabilities.

## Command line

```bash
python -m exam_digitizer.enrichment.cli booklet.pdf --booklet-id T001 --out analyzed.json
python -m exam_digitizer.enrichment.cli page1.png --no-ner --out figures.json
```

Use `--ner-model MODEL_OR_DIRECTORY` to load another compatible trained spaCy model.

## Figure detector limits

The detector identifies candidate large two-dimensional ink regions using geometry
and connected strokes, after suppressing page rulings and headers/footers. It can
locate common connected boxes, circles, flowcharts, axes and curves. It does not
understand the content of a drawing or reliably distinguish diagrams, photographs,
tables and unusually large writing. Small or disconnected drawings may be missed;
text touching a drawing may be included in its box. Every region requires review.

Detection adds annotations after the existing OCR step. It does not remove drawings
from the original OCR input or change answer boundaries/transcription. The existing
answer confidence calibration has not been validated for figure-heavy booklets.
The previous CER/WER and segmentation results do not measure these new features.

## Files and checks

| File under `src/exam_digitizer/enrichment/` | Responsibility |
|---|---|
| `ner.py` | Lazy local NER model, entity labels and exact character offsets |
| `figures.py` | Candidate figure boxes and heuristic review scores |
| `analyzer.py` | Compose annotation with an existing digitizer/result |
| `api.py` | Optional app with retained original routes and new analysis routes |
| `cli.py` | Optional analysis command |

```bash
python -m pytest tests/test_enrichment.py tests/test_enrichment_api.py
```

Tests use synthetic drawings, negative text/ruling examples, Unicode offsets,
injected OCR, PDF input, API compatibility and missing-model handling. The real
English NER model test runs when that model is installed. No test downloads models.

Library references: [spaCy model installation](https://spacy.io/usage/models),
[spaCy entity offsets](https://spacy.io/usage/spacy-101),
[OpenCV connected components](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html).
