# Python application file guide

The application code is organized under `src/exam_digitizer/`. It runs without a
notebook or notebook conversion tools. The original application logic, constants,
model choices, algorithms and output fields are unchanged.

## Use these files for the product

Install from the repository root using Python 3.10 or newer:

```bash
python -m pip install -e ".[api,pdf]"
```

Use the existing public interface when integrating with other people's code:

```python
from exam_digitizer import Config, Digitizer

digitizer = Digitizer(Config(device="cpu"))  # Create once; reuse for booklets.
result = digitizer.digitize(
    ["page1.png", "page2.png"],
    booklet_id="T001",
    roster=["23D001", "23D019"],
)
```

For a service that other applications call:

```bash
python -m exam_digitizer.cli serve --port 8000
```

Open `http://127.0.0.1:8000/docs` for the HTTP contract or
`http://127.0.0.1:8000/ui` for the review page. The existing [API reference](API.md)
and [integration guide](integration.md) describe request/response fields and partner
service integration. No notebook execution is required to use the product.

| File | Responsibility / where to investigate an issue |
|---|---|
| `src/exam_digitizer/digitizer.py` | Pipeline orchestration and result assembly |
| `src/exam_digitizer/config.py` | Device, model, roster and review settings |
| `src/exam_digitizer/schema.py` | Shared data contract and validation |
| `src/exam_digitizer/preprocessing/page.py` | Loading, PDF rendering, deskewing, page cleanup |
| `src/exam_digitizer/models/layout.py` | Text-line boxes and margin marks |
| `src/exam_digitizer/models/glyphs.py` | Glyph CNN, roll numbers and question numbers |
| `src/exam_digitizer/models/recognition.py` | TrOCR and Tesseract line recognition |
| `src/exam_digitizer/models/structuring.py` | Grouping recognized text by question |
| `src/exam_digitizer/models/confidence.py` | Calibrated confidence and review routing |
| `src/exam_digitizer/evaluation/metrics.py` | CER, WER and other evaluation metrics |
| `src/exam_digitizer/api/app.py` | HTTP routes, uploads, limits and preview |
| `src/exam_digitizer/api/review.html` | Human review interface |
| `src/exam_digitizer/cli.py` | Command-line entry point |
| `src/exam_digitizer/enrichment/` | Optional NER, figure detection and separate analysis interfaces |
| `demo/exam_pipeline.py` | Digitizer → guard → grader integration |
| `demo/partner_standins.py` | Example partner service contracts |
| `tests/` | Existing unit, API and integration checks |

## Tests

For the optional analysis layer, see [NER and figure detection](enrichment.md).

Install the development dependencies and run the application tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

To skip tests that load OCR models, set `EXAM_DIGITIZER_SKIP_SLOW=1` in your shell
before running pytest. The remaining tests still require the application's Python
dependencies.

When investigating an issue, use the module table above to locate the responsible
stage and the filename/line reported by the Python traceback.
