"""
Tests for the /ready readiness probe endpoint.
"""

import app.main as main_module
import pytest
from app.main import app
from fastapi.testclient import TestClient

# Fixtures (mirrors test_api.py setup)


class MockModel:
    classes = ["sci.space", "rec.sport.hockey", "talk.politics.guns"]

    def predict(self, text: str) -> tuple[str, float]:
        return self.classes[0], 0.87


class SlowModel:
    """Simulates a model whose inference exceeds the 500ms threshold."""

    classes = ["sci.space"]

    def predict(self, text: str) -> tuple[str, float]:
        import time
        time.sleep(0.6)  # 600ms — over budget
        return self.classes[0], 0.5


class BrokenModel:
    """Simulates a model that raises during inference."""

    def predict(self, text: str) -> tuple[str, float]:
        raise RuntimeError("ONNX runtime failure")


@pytest.fixture(autouse=True)
def inject_mock_model(monkeypatch):
    monkeypatch.setattr(main_module, "_model", MockModel())
    monkeypatch.setattr(main_module, "_model_backend", "sklearn")
    yield
    monkeypatch.setattr(main_module, "_model", None)
    monkeypatch.setattr(main_module, "_model_backend", "none")


client = TestClient(app)

# /ready — happy path


def test_ready_returns_200():
    response = client.get("/ready")
    assert response.status_code == 200


def test_ready_response_shape():
    response = client.get("/ready")
    body = response.json()
    assert "status" in body
    assert "model_backend" in body
    assert "inference_ms" in body


def test_ready_status_is_ready():
    response = client.get("/ready")
    assert response.json()["status"] == "ready"


def test_ready_reports_correct_backend():
    response = client.get("/ready")
    assert response.json()["model_backend"] == "sklearn"


def test_ready_inference_ms_is_positive():
    response = client.get("/ready")
    assert response.json()["inference_ms"] >= 0

# /ready — model not loaded


def test_ready_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main_module, "_model", None)
    response = client.get("/ready")
    assert response.status_code == 503


def test_ready_503_detail_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main_module, "_model", None)
    response = client.get("/ready")
    assert "not loaded" in response.json()["detail"]


# /ready — inference errors


def test_ready_503_when_model_raises(monkeypatch):
    monkeypatch.setattr(main_module, "_model", BrokenModel())
    response = client.get("/ready")
    assert response.status_code == 503


def test_ready_503_detail_when_model_raises(monkeypatch):
    monkeypatch.setattr(main_module, "_model", BrokenModel())
    response = client.get("/ready")
    assert "Inference error" in response.json()["detail"]


# /ready : slow inference (over 500ms budget)


def test_ready_503_when_inference_too_slow(monkeypatch):
    monkeypatch.setattr(main_module, "_model", SlowModel())
    response = client.get("/ready")
    assert response.status_code == 503


def test_ready_503_detail_mentions_timing(monkeypatch):
    monkeypatch.setattr(main_module, "_model", SlowModel())
    response = client.get("/ready")
    assert "too slow" in response.json()["detail"]