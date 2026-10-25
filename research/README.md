# Complete notebook research code

This directory preserves **all 97 code cells** from the original project notebook,
including dataset generation, training, experiments, comparisons, evaluation,
packaging and integration demonstrations. Its 71 markdown cells are retained as
comments alongside the code. The original notebook has also been restored from Git
at [`../notebooks/snlp_group1_week7.ipynb`](../notebooks/snlp_group1_week7.ipynb).

The current application, its new NER/figure features, its tests and its data files
are separate. No application or data files were changed by this restoration.
Use `src/exam_digitizer/` for product development; use this directory to reproduce
the original research workflow.

## Organization

| Location | Contents |
|---|---|
| `stages/00_setup.py` – `stages/04_metrics_and_baseline.py` | Setup, architecture, data download/generation, preprocessing and baseline |
| `stages/05_layout.py` – `stages/07_recognition_and_structuring.py` | Layout, glyph CNN training, TrOCR and question segmentation |
| `stages/08_model_comparisons.py` – `stages/09_calibration.py` | Ablations, recorded cloud responses, LLM/VLM comparisons and calibration |
| `stages/10_test_evaluation.py` – `stages/15_confidence_and_export.py` | Frozen-test evaluation, line benchmark, error analysis, robustness, embedded assets and export |
| `stages/16_package_generation.py` – `stages/19_package_docs.py` | Historical package generation, original tests/assets, regression, API/CLI and docs |
| `stages/20_platform_and_review_api.py` – `stages/25_integration_docs_and_submission.py` | Platform integration, review UI, partner services, marks, responsible-AI checks and submission |
| `file_cells/cell_NNN/` | Exact original file payloads, readable as `.py`, `.md`, `.html`, `.json`, etc. |
| `source_map.json` | Every cell's script, line span, source checksum and file-payload locations |
| `export_notebook.py` | Reproducible converter and read-only completeness check |
| `run.py` | Ordered research runner with separate output directories |
| `tests/test_restoration.py` | Preservation, syntax, shared-state and output-isolation checks |

Every stage has `# %%` markers naming the original notebook cell. Indices start at
zero and count both markdown and code cells. The original Python statements,
constants, model choices and embedded data strings are preserved. Existing notebook
outputs/plots are retained in the restored notebook; executing the research scripts
recomputes them.

Notebook-only `%%writefile` directives call a small Python file-writing helper.
Their original contents live under `file_cells/`. Both Week-6 and Week-7 versions
are kept when the notebook writes a file more than once. Code embedded inside the
`_SMALL` and `_TESTS` string dictionaries is also available as readable files;
the original dictionary statements remain unchanged in their stages.

These historical package snapshots reproduce the original notebook. They do not
include the NER/figure extensions added later to the current application.

## Verify completeness without running experiments

Run from the project root (only the Python standard library is needed):

```bash
python research/export_notebook.py --check
python -m unittest discover -s research/tests -v
```

The checks cover every cell, exact source lines/AST, original generated file bodies,
string-embedded modules/tests, valid Python syntax, shared function globals,
tracebacks and isolated output handling. They never run the original experiments,
download models, train networks or write to the existing application/data.

To regenerate the research export after intentionally editing the notebook:

```bash
python research/export_notebook.py
```

This writes only generated files within `research/`. Generated stages/snapshots are
replaced on regeneration; keep intentional research changes in the notebook and
regenerate, or maintain your own separate experiment files.

## Run the original research

List stages without executing anything or loading models:

```bash
python research/run.py --list
```

Install research dependencies, then run from setup through the selected stage:

```bash
python -m pip install -r research/requirements.txt
python research/run.py --through 07_recognition_and_structuring
```

To reproduce the full notebook:

```bash
python research/run.py --through all
```

The runner keeps one shared Python namespace, matching the notebook's execution
order. Later stages require the objects created in earlier stages, so execute them
through the runner. A new process starts from setup; it does not resume a previous
kernel's state. Tracebacks point to the actual research script and line; use the
nearest cell marker or `source_map.json` to locate the original cell.

## Where research writes its outputs

Each explicit execution gets a new directory:

```text
research/runs/<timestamp>/
├── work/          original research data, caches, results and figures
└── repository/    independently generated historical application/demo package
```

Use `--run-dir experiment_name` to choose a new folder under `research/runs/`.
The runner overrides inherited `G1_WORKDIR` and `G1_REPO` values and changes into the
run directory before executing notebook statements. It rejects paths outside that
area and refuses to reuse a nonempty directory. The notebook's file-generation and
archive steps operate on that reproduced repository, preserving current product
source and data. The run directories are ignored by Git.

The original Colab/Linux assumptions remain: setup can call `apt-get` if Tesseract
is missing, later demo cells use `/tmp`, and the full browser demo optionally needs
Playwright and Chromium. Reproduce the complete workflow in Linux/WSL with Tesseract
and `/tmp` available. The original `/tmp` demo outputs and third-party download
caches still use their original locations. The runner does not rewrite those
statements or emulate the operating system.

Research execution downloads datasets/models, trains a glyph CNN and performs
lengthy comparisons. Original cloud replay/live-call settings are retained in
`stages/00_setup.py`, with recorded responses in the comparison stage. No research
execution or dependency installation is needed for the current product application.
