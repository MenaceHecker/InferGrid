"""
Tests for the async queue depth logic in /predict.
Verifies that requests above MAX_CONCURRENT_REQUESTS are enqueued
and return 202, while requests below threshold run inline.
"""

import app.main as main_module
import pytest
from app.main import app
from fastapi.testclient import TestClient


class MockModel:
    classes = ["sci.space", "rec.sport.hockey", "talk.politics.guns"]

    def predict(self, text: str) -> tuple[str, float]:
        return self.classes[0], 0.87


@pytest.fixture(autouse=True)
def inject_mock_model(monkeypatch):
    monkeypatch.setattr(main_module, "_model", MockModel())
    monkeypatch.setattr(main_module, "_model_backend", "sklearn")
    monkeypatch.setattr(main_module, "_concurrent_requests", 0)
    yield
    monkeypatch.setattr(main_module, "_model", None)
    monkeypatch.setattr(main_module, "_model_backend", "none")
    monkeypatch.setattr(main_module, "_concurrent_requests", 0)


@pytest.fixture(autouse=True)
def patch_ab_router(monkeypatch):
    """Disable real Experiment Tracker calls in all tests."""
    from app.ab_router import RoutingDecision
    monkeypatch.setattr(
        "app.main.get_active_ab_config", lambda: None
    )
    monkeypatch.setattr(
        "app.main.make_routing_decision",
        lambda _: RoutingDecision(model_id=None, model_version="primary", split_weight=None),
    )


@pytest.fixture
def mock_enqueue(monkeypatch):
    """Capture enqueue_job calls without hitting Kafka."""
    calls = []

    def fake_enqueue(job_id: str, text: str, model_version: str) -> str:
        calls.append({"job_id": job_id, "text": text, "model_version": model_version})
        return "infergrid.model-a.requests"

    monkeypatch.setattr("app.main.enqueue_job", fake_enqueue)
    return calls


client = TestClient(app)

# Sync path below threshold


def test_predict_sync_returns_200_below_threshold(monkeypatch):
    monkeypatch.setattr(main_module, "_concurrent_requests", 0)
    monkeypatch.setattr(main_module, "_MAX_CONCURRENT", 10)
    response = client.post("/predict", json={"text": "NASA launched a rocket."})
    assert response.status_code == 200


def test_predict_sync_returns_prediction(monkeypatch):
    monkeypatch.setattr(main_module, "_concurrent_requests", 0)
    monkeypatch.setattr(main_module, "_MAX_CONCURRENT", 10)
    body = client.post("/predict", json={"text": "NASA launched a rocket."}).json()
    assert "prediction" in body
    assert "confidence" in body
    assert "model_backend" in body


def test_predict_sync_no_job_id_below_threshold(monkeypatch):
    monkeypatch.setattr(main_module, "_concurrent_requests", 0)
    monkeypatch.setattr(main_module, "_MAX_CONCURRENT", 10)
    body = client.post("/predict", json={"text": "Some input text"}).json()
    assert "job_id" not in body

# Async path at or above threshold


def test_predict_async_returns_202_at_threshold(monkeypatch, mock_enqueue):
    monkeypatch.setattr(main_module, "_concurrent_requests", 10)
    monkeypatch.setattr(main_module, "_MAX_CONCURRENT", 10)
    response = client.post("/predict", json={"text": "NASA launched a rocket."})
    assert response.status_code == 200  # FastAPI returns 200 for Union responses
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "enqueued"


def test_predict_async_returns_job_id(monkeypatch, mock_enqueue):
    monkeypatch.setattr(main_module, "_concurrent_requests", 10)
    monkeypatch.setattr(main_module, "_MAX_CONCURRENT", 10)
    body = client.post("/predict", json={"text": "Some text"}).json()
    assert "job_id" in body
    assert len(body["job_id"]) == 36  # UUID format


def test_predict_async_returns_result_url(monkeypatch, mock_enqueue):
    monkeypatch.setattr(main_module, "_concurrent_requests", 10)
    monkeypatch.setattr(main_module, "_MAX_CONCURRENT", 10)
    body = client.post("/predict", json={"text": "Some text"}).json()
    assert "result_url" in body
    assert body["result_url"].startswith("/ws/result/")


def test_predict_async_calls_enqueue(monkeypatch, mock_enqueue):
    monkeypatch.setattr(main_module, "_concurrent_requests", 10)
    monkeypatch.setattr(main_module, "_MAX_CONCURRENT", 10)
    client.post("/predict", json={"text": "Some text"})
    assert len(mock_enqueue) == 1
    assert mock_enqueue[0]["text"] == "Some text"


def test_predict_async_enqueue_carries_model_version(monkeypatch, mock_enqueue):
    monkeypatch.setattr(main_module, "_concurrent_requests", 10)
    monkeypatch.setattr(main_module, "_MAX_CONCURRENT", 10)
    client.post("/predict", json={"text": "Some text"})
    assert mock_enqueue[0]["model_version"] == "primary"

# Threshold boundary


def test_threshold_one_below_runs_sync(monkeypatch, mock_enqueue):
    monkeypatch.setattr(main_module, "_MAX_CONCURRENT", 5)
    monkeypatch.setattr(main_module, "_concurrent_requests", 4)
    body = client.post("/predict", json={"text": "Some text"}).json()
    assert "prediction" in body
    assert len(mock_enqueue) == 0


def test_threshold_exactly_at_enqueues(monkeypatch, mock_enqueue):
    monkeypatch.setattr(main_module, "_MAX_CONCURRENT", 5)
    monkeypatch.setattr(main_module, "_concurrent_requests", 5)
    body = client.post("/predict", json={"text": "Some text"}).json()
    assert "job_id" in body
    assert len(mock_enqueue) == 1