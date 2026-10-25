# Handwritten Examination Digitizer — Group 1 (NLP Course Project 2026)

**Asset:** scanned handwritten answer booklet → question-wise answer text (JSON) with a calibrated confidence and a
`needs_review` flag, plus a browser page on which a human checks and corrects what was read. A shared OCR service of the
*AI University* platform; its output is the input of Group 2's AI Exam Grader.

```
page images / PDF ──► preprocessing ──► layout analysis ──► TrOCR (lines) + glyph CNN (roll no., Q-numbers)
                      normalise·deskew   lines · margin marks   constrained decoding · roster check
                   ──► question segmentation ──► calibrated confidence ──► JSON (schema 1.0)
platform flow (Week 7):  scan ─► G1 /digitize ─► human review page (low confidence) ─► G12 input guard ─► G2 grader ─► marksheet
```

Team: J001, J002, J004, J005, J006 · Asset version 0.7.0 (Week 7) · Output schema 1.0

## The "reusable asset" test — what another group needs tomorrow

| | |
|---|---|
| **Component** | Handwritten Examination Digitizer |
| **Input** | one booklet = page images (PNG/JPEG, in order) or one PDF; optional class roster (valid roll numbers) |
| **Output** | `{booklet_id, student_id, student_id_confidence, answers:[{student_id, question, answer_text, page, confidence, needs_review, flags}], pages, model, runtime_s}` |
| **Interface** | Python: `Digitizer().digitize(pages)` · CLI: `exam-digitizer digitize p1.png p2.png` · HTTP: `POST /digitize` · browser: `GET /ui` |
| **Runs** | locally on CPU (≈3 s/page) or GPU; no data leaves the machine; no API keys; no network after installation |
| **Confidence** | `confidence` = calibrated P(transcript usable: right question label and CER ≤ 10 %); `needs_review` = route to a human |
| **Platform glue** | `demo/exam_pipeline.py`: digitizer → (Group 12 guard) → Group 2 grader, contracts only — see [docs/integration.md](docs/integration.md) |

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

Full field reference, flag vocabulary, HTTP examples, review page and error behaviour: **[docs/API.md](docs/API.md)**.

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

## Run the API and the review page

```bash
python -m exam_digitizer.cli serve --port 8000     # = uvicorn exam_digitizer.api.app:app --port 8000
# browser: http://127.0.0.1:8000/ui  → upload the pages + roster → check the boxed lines → correct → approve → export JSON
curl -s -F "pages=@T001_p1.png" -F "pages=@T001_p2.png" -F "booklet_id=T001" -F "roster=23D001,23D019" \
     http://127.0.0.1:8000/digitize | python -m json.tool
```

`GET /health`, `GET /schema` (JSON schema + example), `GET /docs` (interactive OpenAPI UI), `POST /digitize-base64`
(JSON body, for service-to-service calls), `POST /preview` (the deskewed pages the line boxes refer to). A server-wide
roster can be set with `EXAM_DIGITIZER_ROSTER=/path/roster.txt`; request limits with `EXAM_DIGITIZER_MAX_PAGES` (40) and
`EXAM_DIGITIZER_MAX_MB` (60). There is no authentication: keep the default 127.0.0.1 binding or put the service behind
the university's reverse proxy / SSO.

![review page](docs/review_ui.png)

## Use it in the AI University platform (Week 7)

```bash
python -m exam_digitizer.cli serve --port 8000          # Group 1 (this asset)
python demo/partner_standins.py --port 8100             # stand-ins for Group 2's grader and Group 12's guard (contracts from the handout)
python demo/exam_pipeline.py demo/sample_booklet/T001_p1.png --answer-key demo/answer_key.json --roster demo/roster.txt \
       --digitizer-url http://127.0.0.1:8000 --grader-url http://127.0.0.1:8100/evaluate-answer \
       --guard-url http://127.0.0.1:8100/check-input
python demo/exam_pipeline.py demo/T001_reviewed.json --answer-key demo/answer_key.json   # a booklet corrected on the review page
```

Answers are graded automatically only when the digitizer does not ask for review (or a human approved them), the guard
calls them safe and the student is identified; everything else lands in a review queue with the reason. Replace the
stand-in URLs by Group 2's / Group 12's services when they are deployed — nothing else changes.
Placement in the platform, consumers and contracts: **[docs/integration.md](docs/integration.md)**.

![platform](docs/ai_university_architecture.png)

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
│   ├── api/app.py            FastAPI service (+ /ui, /preview, request limits)
│   ├── api/review.html       the human-review page
│   ├── cli.py                command line
│   └── assets/               glyph_cnn.pt · calibration.json
├── tests/                    pytest (unit + integration tests run without models; end-to-end + API tests need TrOCR)
├── notebooks/                the weekly project notebook (data, experiments, evaluation, packaging, integration)
├── docs/                     API.md · integration.md · responsible_ai.md · evaluation.md · architecture.png ·
│                             ai_university_architecture.png · review_ui.png
├── demo/                     sample booklet + roster + answer key · exam_pipeline.py · partner_standins.py ·
│                             group2_consumer.py · reviewed booklet and marksheet · curl example
└── results/                  frozen test-set results, Week-6 regression check, Week-7 integration evidence
```

## How well does it work? (frozen test set, 40 booklets / 114 answers; Week 5)

| System | answer CER ↓ | label-free CER ↓ | question-seg. F1 ↑ | student-ID acc. / wrong | s/page (CPU) |
|---|---|---|---|---|---|
| Baseline: Tesseract + regex | 0.939 | 0.443 | 0.164 | 0.00 / 0.00 | 3.1 |
| **This asset: layout + TrOCR-small** | **0.124** [0.070, 0.188] | **0.045** | **0.925** | **0.95 / 0.00** | 2.8 |

Two-thirds of the remaining error is answers filed under a misread question number; confidence routes most of those
to review (accept the 70 % most confident answers automatically → 5 % unusable, vs 19.5 % without review).
**In the platform (Week 7):** graded by a strict keyword grader, digitization costs 1.5 of 10 marks per answer with the
default review policy (17 % of answers re-read by staff) and 0.7 at review threshold 0.8 (37 % re-read); the routed
answers are the ones that would have lost most (5.2 marks vs 1.8) — [docs/integration.md](docs/integration.md).
Details, error analysis and robustness: [docs/evaluation.md](docs/evaluation.md) and the notebook.

## Tests

```bash
pytest                               # unit + integration tests (seconds) + end-to-end/API tests (need TrOCR; skipped if unavailable)
EXAM_DIGITIZER_SKIP_SLOW=1 pytest    # without the model tests
```

## Responsible use — summary of [docs/responsible_ai.md](docs/responsible_ai.md)

* Runs fully locally and works with the network blocked; stores nothing. Never send real student scripts to cloud
  OCR/LLM services (assignment §8) — the Week-4 cloud systems are not part of the asset.
* The system is an **assistant**, not a decision maker: every answer carries `confidence`, `needs_review` and `flags`;
  low-confidence answers and unidentified students must go to a human, never be auto-graded. The review page records
  who approved or corrected what.
* Known blind spot: fluent wrong words with high confidence (*footrope → footage*): 2.0 per 100 words on lines the
  model is ≥ 0.9 sure of. Graders must match rubric points tolerantly; staff should spot-check accepted answers.
* Bias: no significant effect of handwriting width, stroke or slant on 40 writers; error concentrates on a few writers,
  and so does review (62 % vs 11 % of answers routed). Must be re-checked on real anonymised booklets.
* Student writing is untrusted: route `answer_text` through Group 12's guard before any LLM. The API has no
  authentication — keep it on 127.0.0.1 or behind the university's SSO.
* Known limits: English Latin-script prose only; question numbers in the left margin of a ruled answer sheet; no
  diagrams/tables; roll-number policy `digits-exact` returns `null` rather than guessing (1 in 8 off-roster students was
  still mis-assigned in the stress test).