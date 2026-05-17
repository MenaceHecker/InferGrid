"""
Tests for Prometheus metric instrumentation.
Verifies that request_count, request_latency, and prediction_confidence
are recorded correctly after API calls.
"""

import app.main as main_module
import pytest
from app.main import app
from app.metrics import PREDICTION_CONFIDENCE, REQUEST_COUNT, REQUEST_LATENCY
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY


class MockModel:
    classes = ["sci.space", "rec.sport.hockey", "talk.politics.guns"]

    def predict(self, text: str) -> tuple[str, float]:
        return self.classes[0], 0.87


@pytest.fixture(autouse=True)
def inject_mock_model(monkeypatch):
    monkeypatch.setattr(main_module, "_model", MockModel())
    monkeypatch.setattr(main_module, "_model_backend", "sklearn")
    yield
    monkeypatch.setattr(main_module, "_model", None)
    monkeypatch.setattr(main_module, "_model_backend", "none")


client = TestClient(app)


def _get_counter_value(counter, labels: dict[str, str]) -> float:
    """Read the current value of a labelled counter sample."""
    label_values = tuple(labels[k] for k in sorted(labels))
    for metric in REGISTRY.collect():
        if metric.name == counter._name:
            for sample in metric.samples:
                if sample.name.endswith("_total"):
                    sample_values = tuple(sample.labels[k] for k in sorted(labels))
                    if sample_values == label_values:
                        return sample.value
    return 0.0


def _get_histogram_count(
    histogram,
    label_key: str | None = None,
    label_value: str | None = None,
) -> float:
    """Read the _count sample of a histogram."""
    for metric in REGISTRY.collect():
        if metric.name == histogram._name:
            for sample in metric.samples:
                if not sample.name.endswith("_count"):
                    continue
                if label_key is None or sample.labels.get(label_key) == label_value:
                    return sample.value
    return 0.0


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_200():
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_endpoint_returns_prometheus_text_format():
    response = client.get("/metrics")
    assert "text/plain" in response.headers["content-type"]


def test_metrics_endpoint_exposes_request_count():
    client.get("/health")
    response = client.get("/metrics")
    assert "http_requests_total" in response.text


def test_metrics_endpoint_exposes_latency():
    client.get("/health")
    response = client.get("/metrics")
    assert "request_latency_seconds" in response.text


def test_metrics_endpoint_exposes_confidence():
    client.post("/predict", json={"text": "NASA launched a rocket."})
    response = client.get("/metrics")
    assert "prediction_confidence" in response.text


def test_metrics_endpoint_exposes_model_load_time():
    response = client.get("/metrics")
    assert "model_load_time_seconds" in response.text


# ---------------------------------------------------------------------------
# request_count increments
# ---------------------------------------------------------------------------


def test_request_count_increments_on_health():
    labels = {"method": "GET", "endpoint": "/health", "status_code": "200"}
    before = _get_counter_value(REQUEST_COUNT, labels)
    client.get("/health")
    after = _get_counter_value(REQUEST_COUNT, labels)
    assert after == before + 1


def test_request_count_increments_on_predict():
    labels = {"method": "POST", "endpoint": "/predict", "status_code": "200"}
    before = _get_counter_value(REQUEST_COUNT, labels)
    client.post("/predict", json={"text": "Some input text"})
    after = _get_counter_value(REQUEST_COUNT, labels)
    assert after == before + 1


def test_request_count_records_422_on_invalid_input():
    labels = {"method": "POST", "endpoint": "/predict", "status_code": "422"}
    before = _get_counter_value(REQUEST_COUNT, labels)
    client.post("/predict", json={"text": ""})
    after = _get_counter_value(REQUEST_COUNT, labels)
    assert after == before + 1


# ---------------------------------------------------------------------------
# request_latency recorded
# ---------------------------------------------------------------------------


def test_latency_histogram_records_health_requests():
    before = _get_histogram_count(REQUEST_LATENCY, "endpoint", "/health")
    client.get("/health")
    after = _get_histogram_count(REQUEST_LATENCY, "endpoint", "/health")
    assert after == before + 1


def test_latency_histogram_records_predict_requests():
    before = _get_histogram_count(REQUEST_LATENCY, "endpoint", "/predict")
    client.post("/predict", json={"text": "Some input text"})
    after = _get_histogram_count(REQUEST_LATENCY, "endpoint", "/predict")
    assert after == before + 1


# ---------------------------------------------------------------------------
# prediction_confidence recorded
# ---------------------------------------------------------------------------


def test_confidence_histogram_increments_on_predict():
    before = _get_histogram_count(PREDICTION_CONFIDENCE)
    client.post("/predict", json={"text": "Some input text"})
    after = _get_histogram_count(PREDICTION_CONFIDENCE)
    assert after == before + 1


def test_confidence_not_recorded_on_invalid_input():
    before = _get_histogram_count(PREDICTION_CONFIDENCE)
    client.post("/predict", json={"text": ""})
    after = _get_histogram_count(PREDICTION_CONFIDENCE)
    assert after == before
