"""
Tests for API key authentication.
Verifies that protected endpoints reject missing/invalid keys
and that exempt endpoints (health, ready, metrics) remain accessible.
"""

import app.main as main_module
import pytest
from app.main import app
from fastapi.testclient import TestClient


class MockModel:
    classes = ["sci.space", "rec.sport.hockey"]

    def predict(self, text: str) -> tuple[str, float]:
        return self.classes[0], 0.91


VALID_KEY = "test-api-key-abc123"


@pytest.fixture(autouse=True)
def inject_mock_model(monkeypatch):
    monkeypatch.setattr(main_module, "_model", MockModel())
    monkeypatch.setattr(main_module, "_model_backend", "sklearn")
    monkeypatch.setattr(main_module, "_concurrent_requests", 0)
    yield
    monkeypatch.setattr(main_module, "_model", None)
    monkeypatch.setattr(main_module, "_model_backend", "none")


@pytest.fixture(autouse=True)
def patch_ab_router(monkeypatch):
    from app.ab_router import RoutingDecision

    monkeypatch.setattr("app.main.get_active_ab_config", lambda: None)
    monkeypatch.setattr(
        "app.main.make_routing_decision",
        lambda _: RoutingDecision(model_id=None, model_version="primary", split_weight=None),
    )


@pytest.fixture
def client_with_key(monkeypatch):
    """Client with API_KEY env var set."""
    import app.auth as auth_module

    monkeypatch.setattr(auth_module, "API_KEY", VALID_KEY)
    return TestClient(app)


@pytest.fixture
def client_no_key(monkeypatch):
    """Client simulating environment with no API_KEY configured."""
    import app.auth as auth_module

    monkeypatch.setattr(auth_module, "API_KEY", "")
    return TestClient(app)


# Exempt endpoints - no auth required


def test_health_no_key_returns_200(client_with_key):
    response = client_with_key.get("/health")
    assert response.status_code == 200


def test_ready_no_key_returns_200(client_with_key):
    response = client_with_key.get("/ready")
    assert response.status_code == 200


def test_metrics_no_key_returns_200(client_with_key):
    response = client_with_key.get("/metrics")
    assert response.status_code == 200


# /predict - auth required


def test_predict_with_valid_key_returns_200(client_with_key):
    response = client_with_key.post(
        "/predict",
        json={"text": "NASA launched a rocket."},
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 200


def test_predict_missing_key_returns_401(client_with_key):
    response = client_with_key.post(
        "/predict",
        json={"text": "NASA launched a rocket."},
    )
    assert response.status_code == 401


def test_predict_wrong_key_returns_403(client_with_key):
    response = client_with_key.post(
        "/predict",
        json={"text": "NASA launched a rocket."},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 403


def test_predict_missing_key_detail(client_with_key):
    response = client_with_key.post(
        "/predict",
        json={"text": "NASA launched a rocket."},
    )
    assert "Missing" in response.json()["detail"]


def test_predict_wrong_key_detail(client_with_key):
    response = client_with_key.post(
        "/predict",
        json={"text": "NASA launched a rocket."},
        headers={"X-API-Key": "wrong-key"},
    )
    assert "Invalid" in response.json()["detail"]


# Auth disabled when API_KEY not set (local dev)


def test_predict_no_env_key_allows_request(client_no_key):
    """When API_KEY is not configured, auth is skipped with a warning."""
    response = client_no_key.post(
        "/predict",
        json={"text": "NASA launched a rocket."},
    )
    assert response.status_code == 200


def test_predict_no_env_key_ignores_header(client_no_key):
    """Any key value is accepted when API_KEY env var is not set."""
    response = client_no_key.post(
        "/predict",
        json={"text": "NASA launched a rocket."},
        headers={"X-API-Key": "anything"},
    )
    assert response.status_code == 200
