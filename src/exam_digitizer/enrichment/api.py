"""Optional API: uvicorn exam_digitizer.enrichment.api:app --port 8000.

All original routes are retained. POST /analyze and /analyze-base64 return a
separate envelope containing unchanged digitization plus NER/figure annotations.
"""
from __future__ import annotations

import base64
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..api import app as original
from ..schema import DigitizeBase64Request, DigitizeResult
from .analyzer import DocumentAnalyzer
from .figures import FigureDetector
from .ner import NERUnavailableError, SpacyNER


class Entity(BaseModel):
    answer_index: int = Field(..., ge=0)
    text: str
    label: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class Figure(BaseModel):
    page: int = Field(..., ge=1)
    bbox: List[int] = Field(..., min_length=4, max_length=4)
    label: str
    score: float = Field(..., ge=0, le=1, description="Heuristic ranking, not calibrated probability")
    score_kind: str
    needs_review: bool


class Analysis(BaseModel):
    schema_version: str = "1.0"
    entities: List[Entity]
    figures: List[Figure]
    ner_enabled: bool
    figure_detection_enabled: bool
    figure_coordinate_space: str


class AnalyzeResult(BaseModel):
    digitization: DigitizeResult
    analysis: Analysis


class AnalyzeRequest(DigitizeBase64Request):
    detect_entities: bool = True
    detect_figures: bool = True


def create_app(*, ner=None, figures=None) -> FastAPI:
    """Build an opt-in app without adding routes to the original app object."""
    app = FastAPI(
        title="Handwritten Examination Digitizer with document analysis",
        version=original.__version__, lifespan=original.lifespan,
        description="Original digitizer routes plus optional local NER and candidate figure detection.",
    )
    app.include_router(original.app.router)
    ner = ner if ner is not None else SpacyNER()
    figures = figures if figures is not None else FigureDetector()

    def run(data, options, detect_entities, detect_figures):
        pages = original._pages(data)  # Reuse the existing upload/PDF limits.
        if options["roll_policy"] not in original.ROLL_POLICIES:
            raise HTTPException(422, f"roll_policy must be one of {original.ROLL_POLICIES}")
        try:
            analyzer = DocumentAnalyzer(original.get_digitizer(), ner=ner, figures=figures)
            return analyzer.analyze(pages, detect_entities=detect_entities,
                                    detect_figures=detect_figures, **options)
        except NERUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/analyze", response_model=AnalyzeResult, response_model_exclude_none=True)
    async def analyze(
        pages: List[UploadFile] = File(...),
        booklet_id: str = Form(""), roster: Optional[str] = Form(None),
        roll_policy: str = Form("digits-exact"), include_lines: bool = Form(False),
        review_threshold: float = Form(0.5), detect_entities: bool = Form(True),
        detect_figures: bool = Form(True),
    ):
        try:
            parsed_roster = original._parse_roster(roster)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, "roster must be comma-separated text or a JSON list") from exc
        return run([await page.read() for page in pages], {
            "booklet_id": booklet_id, "roster": parsed_roster, "roll_policy": roll_policy,
            "include_lines": include_lines, "review_threshold": review_threshold,
        }, detect_entities, detect_figures)

    @app.post("/analyze-base64", response_model=AnalyzeResult, response_model_exclude_none=True)
    def analyze_base64(request: AnalyzeRequest):
        try:
            data = [base64.b64decode(page, validate=True) for page in request.pages]
        except ValueError as exc:
            raise HTTPException(400, "pages must be valid base64 strings") from exc
        options = request.model_dump(exclude={"pages", "detect_entities", "detect_figures"})
        return run(data, options, request.detect_entities, request.detect_figures)

    return app


app = create_app()
