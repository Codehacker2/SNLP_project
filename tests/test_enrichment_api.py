"""New analysis routes remain separate from the original API; no OCR downloads."""
import base64
import copy

import pytest
from fastapi.testclient import TestClient

from exam_digitizer.api import app as original
from exam_digitizer.enrichment.api import create_app
from exam_digitizer.enrichment.ner import NERUnavailableError
from exam_digitizer.schema import EXAMPLE_RESULT


class ExistingDigitizer:
    def digitize(self, pages, **kwargs):
        return copy.deepcopy(EXAMPLE_RESULT)


class TestNER:
    __test__ = False

    def extract_many(self, texts):
        return [[] for _ in texts]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(original, "get_digitizer", lambda: ExistingDigitizer())
    return TestClient(create_app(ner=TestNER()))


def test_new_routes_do_not_modify_original_app(client):
    assert "/analyze" not in {route.path for route in original.app.routes}
    paths = client.get("/openapi.json").json()["paths"]
    assert "/analyze" in paths and "/analyze-base64" in paths and "/digitize" in paths
    assert client.get("/ui").status_code == 200
    assert client.get("/schema").json() == TestClient(original.app).get("/schema").json()


def test_old_digitize_response_is_identical_on_optional_app(client, page_file):
    image = page_file.read_bytes()
    old = TestClient(original.app).post("/digitize", files={"pages": ("p.png", image, "image/png")})
    retained = client.post("/digitize", files={"pages": ("p.png", image, "image/png")})
    assert old.status_code == retained.status_code == 200
    assert old.json() == retained.json()
    assert "analysis" not in retained.json()


def test_multipart_analysis_encloses_the_original_response(client, page_file):
    response = client.post("/analyze", files={"pages": ("p.png", page_file.read_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    assert response.json()["digitization"] == EXAMPLE_RESULT
    assert response.json()["analysis"]["entities"] == []
    assert response.json()["analysis"]["figures"] == []


def test_base64_analysis_and_invalid_base64(client, page_file):
    encoded = base64.b64encode(page_file.read_bytes()).decode("ascii")
    response = client.post("/analyze-base64", json={"pages": [encoded], "detect_figures": False})
    assert response.status_code == 200, response.text
    assert response.json()["digitization"] == EXAMPLE_RESULT
    bad = client.post("/analyze-base64", json={"pages": ["!invalid!"]})
    assert bad.status_code == 400


def test_missing_model_is_reported_and_ner_can_be_disabled(monkeypatch, page_file):
    class UnavailableNER:
        def extract_many(self, texts):
            raise NERUnavailableError("Install the English NER model")

    monkeypatch.setattr(original, "get_digitizer", lambda: ExistingDigitizer())
    client = TestClient(create_app(ner=UnavailableNER()))
    files = {"pages": ("p.png", page_file.read_bytes(), "image/png")}
    assert client.post("/analyze", files=files).status_code == 503
    response = client.post("/analyze", files=files, data={"detect_entities": "false", "detect_figures": "false"})
    assert response.status_code == 200


def test_analysis_keeps_existing_limits(client, monkeypatch):
    monkeypatch.setattr(original, "MAX_PAGES", 1)
    files = [("pages", ("p1.png", b"x", "image/png")), ("pages", ("p2.png", b"x", "image/png"))]
    assert client.post("/analyze", files=files).status_code == 413


def test_bad_roll_policy_and_roster_are_rejected(client, page_file):
    files = {"pages": ("p.png", page_file.read_bytes(), "image/png")}
    assert client.post("/analyze", files=files, data={"roll_policy": "invalid"}).status_code == 422
    assert client.post("/analyze", files=files, data={"roster": "[broken-json"}).status_code == 422
