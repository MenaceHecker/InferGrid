"""
Integration tests for the InferGrid Inference API.
Uses a tiny in-memory mock model so tests run without a real .pkl file.
"""

import app.main as main_module
import pytest
from app.main import app
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockModel:
    """Minimal stand-in for SklearnLoader — no disk I/O required."""

    classes = ["sci.space", "rec.sport.hockey", "talk.politics.guns"]

    def predict(self, text: str) -> tuple[str, float]:
        # Deterministic: return first class with fixed confidence
        return self.classes[0], 0.87


@pytest.fixture(autouse=True)
def inject_mock_model(monkeypatch):
    """Swap the real model for MockModel before every test."""
    monkeypatch.setattr(main_module, "_model", MockModel())
    yield
    monkeypatch.setattr(main_module, "_model", None)


client = TestClient(app)

# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_reports_model_loaded():
    response = client.get("/health")
    assert response.json()["model_loaded"] is True


def test_health_reports_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main_module, "_model", None)
    response = client.get("/health")
    assert response.json()["model_loaded"] is False


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def test_metrics_returns_200():
    response = client.get("/metrics")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /predict — valid inputs
# ---------------------------------------------------------------------------


def test_predict_valid_text():
    response = client.post("/predict", json={"text": "NASA launched a new rocket today."})
    assert response.status_code == 200


def test_predict_response_shape():
    response = client.post("/predict", json={"text": "Some input text"})
    body = response.json()
    assert "prediction" in body
    assert "confidence" in body
    assert "model_backend" in body


def test_predict_confidence_in_range():
    response = client.post("/predict", json={"text": "Some input text"})
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_predict_returns_known_class():
    response = client.post("/predict", json={"text": "Some input text"})
    assert response.json()["prediction"] in MockModel.classes


def test_predict_backend_is_sklearn():
    response = client.post("/predict", json={"text": "Some input text"})
    assert response.json()["model_backend"] == "sklearn"


# ---------------------------------------------------------------------------
# /predict — invalid inputs
# ---------------------------------------------------------------------------


def test_predict_empty_string_returns_422():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_text_field_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_text_too_long_returns_422():
    response = client.post("/predict", json={"text": "a" * 10_001})
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /predict — model not loaded (503)
# ---------------------------------------------------------------------------


def test_predict_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main_module, "_model", None)
    response = client.post("/predict", json={"text": "Some input text"})
    assert response.status_code == 503
