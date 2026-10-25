# Handwritten Examination Digitizer — Group 1 (NLP Course Project 2026)

**Asset:** scanned handwritten answer booklet → question-wise answer text (JSON) with a calibrated confidence and a
`needs_review` flag. Built for the *AI University* platform; its output is the input of Group 2's AI Exam Grader.

```
page images / PDF ──► preprocessing ──► layout analysis ──► TrOCR (lines) + glyph CNN (roll no., Q-numbers)
                      normalise·deskew   lines · margin marks   constrained decoding · roster check
                   ──► question segmentation ──► calibrated confidence ──► JSON (schema 1.0) ──► Group 2 grader
```

Team: J001, J002, J004, J005, J006 · Asset version 0.6.0 (Week 6) · Output schema 1.0

## The "reusable asset" test — what another group needs tomorrow

| | |
|---|---|
| **Component** | Handwritten Examination Digitizer |
| **Input** | one booklet = page images (PNG/JPEG, in order) or one PDF; optional class roster (valid roll numbers) |
| **Output** | `{booklet_id, student_id, student_id_confidence, answers:[{student_id, question, answer_text, page, confidence, needs_review, flags}], pages, model, runtime_s}` |
| **Interface** | Python: `Digitizer().digitize(pages)` · CLI: `exam-digitizer digitize p1.png p2.png` · HTTP: `POST /digitize` |
| **Runs** | locally on CPU (≈3 s/page) or GPU; no data leaves the machine; no API keys |
| **Confidence** | `confidence` = calibrated P(transcript usable: right question label and CER ≤ 10 %); `needs_review` = route to a human |

Example:

```python
from exam_digitizer import Digitizer
dz = Digitizer()                                      # loads TrOCR + glyph CNN once (first run downloads TrOCR, ~250 MB)
result = dz.digitize(["T001_p1.png", "T001_p2.png"], booklet_id="T001", roster=["23D001", "23D002", "23D019"])
for a in result["answers"]:
    print(a["student_id"], a["question"], a["page"], a["confidence"], a["needs_review"], a["answer_text"][:60])
```

```json
{"schema_version": "1.0", "booklet_id": "T001", "student_id": "23D019", "student_id_confidence": 0.999, "pages": 1,
 "answers": [{"student_id": "23D019", "question": "Q4", "page": 1, "confidence": 0.95, "needs_review": false,
              "flags": [], "answer_text": "let us consider first what would appear to be the simplest case ..."}],
 "model": "layout+microsoft/trocr-small-handwritten", "confidence_kind": "calibrated", "runtime_s": 4.45}
```

Full field reference, flag vocabulary, HTTP examples and error behaviour: **[docs/API.md](docs/API.md)**.

## Install

```bash
git clone <this repo> && cd handwritten_exam_digitizer
pip install -e ".[api,pdf]"          # or: pip install -r requirements.txt && export PYTHONPATH=src
python -m exam_digitizer.cli digitize demo/sample_booklet/T001_p1.png --roster demo/roster.txt
```

Python ≥ 3.10. CPU is enough. `tesseract` is **not** required (only for the Week-2 baseline recogniser).
The packaged files `src/exam_digitizer/assets/glyph_cnn.pt` (36-class glyph CNN, 1.6 MB) and
`assets/calibration.json` (Platt scaling) are produced by the project notebook; if `glyph_cnn.pt` is missing the
package re-trains it from MNIST/EMNIST on first use (needs internet, a few minutes).

## Run the API

```bash
python -m exam_digitizer.cli serve --port 8000     # = uvicorn exam_digitizer.api.app:app --port 8000
curl -s -F "pages=@T001_p1.png" -F "pages=@T001_p2.png" -F "booklet_id=T001" -F "roster=23D001,23D019" \
     http://127.0.0.1:8000/digitize | python -m json.tool
```

`GET /health`, `GET /schema` (JSON schema + example), `GET /docs` (interactive OpenAPI UI), `POST /digitize-base64`
(JSON body, for service-to-service calls). A server-wide roster can be set with `EXAM_DIGITIZER_ROSTER=/path/roster.txt`.

## Repository layout

```
├── README.md · requirements.txt · pyproject.toml
├── src/exam_digitizer/
│   ├── digitizer.py          THE ASSET: Digitizer.digitize(pages) -> contract JSON
│   ├── schema.py             pydantic contract (DigitizeResult, Answer), schema 1.0, example
│   ├── config.py             runtime configuration (model, device, roster, policies, thresholds)
│   ├── preprocessing/page.py load · crop-to-paper · normalise · deskew · rule/margin removal · ink mask
│   ├── models/layout.py      text lines + margin marks
│   ├── models/glyphs.py      glyph CNN, question-number lattice decoder, roll-number decoder + roster policies
│   ├── models/recognition.py TrOCR line reader (core) · Tesseract line reader (baseline)
│   ├── models/structuring.py question segmentation, raw confidence, flags
│   ├── models/confidence.py  Platt calibration, review policy
│   ├── evaluation/metrics.py CER/WER, segmentation F1, ID metrics, bootstrap CIs
│   ├── api/app.py            FastAPI service
│   ├── cli.py                command line
│   └── assets/               glyph_cnn.pt · calibration.json
├── tests/                    pytest (unit tests run without models; end-to-end + API tests need TrOCR)
├── notebooks/                the weekly project notebook (data generation, experiments, evaluation, Week-6 packaging)
├── docs/                     API.md · evaluation.md · architecture.png
├── demo/                     sample booklet, roster, Group-2 consumer script, curl example
└── results/                  frozen test-set results, regression check of the packaged asset
```

## How well does it work? (frozen test set, 40 booklets / 114 answers; Week 5)

| System | answer CER ↓ | label-free CER ↓ | question-seg. F1 ↑ | student-ID acc. / wrong | s/page (CPU) |
|---|---|---|---|---|---|
| Baseline: Tesseract + regex | 0.939 | 0.443 | 0.164 | 0.00 / 0.00 | 3.1 |
| **This asset: layout + TrOCR-small** | **0.124** [0.070, 0.188] | **0.045** | **0.925** | **0.95 / 0.00** | 2.8 |

Two-thirds of the remaining error is answers filed under a misread question number; confidence routes most of those
to review (accept the 70 % most confident answers automatically → 5 % unusable, vs 19.5 % without review).
Details, error analysis and robustness: [docs/evaluation.md](docs/evaluation.md) and the notebook.

## Tests

```bash
pytest                               # unit tests (seconds) + end-to-end/API tests (need TrOCR; skipped if unavailable)
EXAM_DIGITIZER_SKIP_SLOW=1 pytest    # unit tests only
```

## Responsible use

* Runs fully locally; never send real student scripts to cloud OCR/LLM services (assignment §8).
* The system is an **assistant**, not a decision maker: every answer carries `confidence`, `needs_review` and `flags`;
  low-confidence answers and unidentified students must go to a human, never be auto-graded.
* Known limits: English Latin-script prose only; expects question numbers in the left margin of a ruled answer sheet;
  no diagrams/tables; roll-number policy `digits-exact` returns `null` rather than guessing (1 in 8 off-roster
  students was still mis-assigned in the stress test); error is unequal across handwriting styles.

## Reusing it inside the AI University platform

Group 2 (grader) consumes `answers[*]` directly (`demo/group2_consumer.py`); any other service that ingests handwritten
documents (lab records, feedback forms) can call the same endpoint — the margin-based question segmentation simply
yields a single `Q?` answer when there are no question numbers.