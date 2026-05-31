"""
tests/test_ab_end_to_end.py

End-to-end Phase 4 test:
  1. Register two model versions
  2. Seed evaluation metrics for both
  3. Configure an A/B split (60/40)
  4. Send 100 /predict requests to the inference API
  5. Verify traffic was split within 10% of configured ratio
  6. Verify /ab/compare returns correct side-by-side metrics

Requires both services running locally:
  - Experiment Tracker: http://localhost:8001
  - Inference API:      http://localhost:8000

Run with:
  pytest tests/test_ab_end_to_end.py -v
  or
  TRACKER_URL=http://... INFERENCE_URL=http://... pytest tests/test_ab_end_to_end.py -v
"""

import hashlib
import os

import pytest
import requests

TRACKER_URL = os.environ.get("TRACKER_URL", "http://localhost:8001")
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8000")

TOTAL_REQUESTS = 100
SPLIT_WEIGHT = 0.6  # 60% model_a, 40% model_b
TOLERANCE = 0.10  # allow ±10% deviation from configured ratio

DATASET_HASH = hashlib.sha256(b"20newsgroups-test-v1").hexdigest()

SAMPLE_TEXTS = [
    "NASA launched a new rocket toward the International Space Station.",
    "The hockey team scored three goals in the final period to win.",
    "Congress is debating new gun control legislation this week.",
    "Scientists discovered a new exoplanet in a nearby star system.",
    "The stock market fell sharply on concerns about inflation.",
    "A new graphics card was released with improved ray tracing.",
    "The motorcycle race was postponed due to rain.",
    "Researchers published findings on a potential cancer treatment.",
    "The baseball team won the championship after twelve innings.",
    "An atheist group filed a lawsuit challenging a local ordinance.",
]


# Helpers


def register_model(name: str, version_tag: str, status: str = "active") -> int:
    version_hash = hashlib.sha256(version_tag.encode()).hexdigest()
    resp = requests.post(
        f"{TRACKER_URL}/models/register",
        json={"name": name, "version_hash": version_hash, "status": status},
        timeout=5,
    )
    # 409 means already registered so fetch the existing id
    if resp.status_code == 409:
        detail = resp.json()["detail"]
        # Extract id from detail string "... (id=N)"
        import re

        match = re.search(r"id=(\d+)", detail)
        if match:
            return int(match.group(1))
    resp.raise_for_status()
    return resp.json()["id"]


def add_metric(model_id: int, metric_name: str, metric_value: float) -> None:
    resp = requests.post(
        f"{TRACKER_URL}/models/{model_id}/evaluations",
        json={
            "metric_name": metric_name,
            "metric_value": metric_value,
            "dataset_hash": DATASET_HASH,
        },
        timeout=5,
    )
    # 409 = metric already seeded, that's fine
    if resp.status_code not in (201, 409):
        resp.raise_for_status()


def configure_ab(model_a_id: int, model_b_id: int, split_weight: float) -> dict:
    resp = requests.post(
        f"{TRACKER_URL}/ab/configure",
        json={
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "split_weight": split_weight,
        },
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()


def send_predict(text: str) -> dict:
    resp = requests.post(
        f"{INFERENCE_URL}/predict",
        json={"text": text},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()


# Fixtures


@pytest.fixture(scope="module")
def registered_models():
    """Register both models once for the whole module."""
    model_a_id = register_model("newsgroups-sklearn-v1", "sklearn-v1", status="active")
    model_b_id = register_model("newsgroups-onnx-v1", "onnx-v1", status="staging")

    add_metric(model_a_id, "accuracy", 0.912)
    add_metric(model_a_id, "f1", 0.897)
    add_metric(model_a_id, "latency_p95_ms", 42.3)

    add_metric(model_b_id, "accuracy", 0.934)
    add_metric(model_b_id, "f1", 0.921)
    add_metric(model_b_id, "latency_p95_ms", 38.7)

    return model_a_id, model_b_id


@pytest.fixture(scope="module")
def ab_config(registered_models):
    """Configure A/B split once for the whole module."""
    model_a_id, model_b_id = registered_models
    return configure_ab(model_a_id, model_b_id, SPLIT_WEIGHT)


@pytest.fixture(scope="module")
def predict_results(ab_config):
    """Send 100 predict requests and collect model_version from each response."""
    texts = (SAMPLE_TEXTS * (TOTAL_REQUESTS // len(SAMPLE_TEXTS) + 1))[:TOTAL_REQUESTS]
    results = [send_predict(text) for text in texts]
    return results


# Step 1 & 2: Model registration and metrics


def test_models_registered(registered_models):
    model_a_id, model_b_id = registered_models
    assert model_a_id > 0
    assert model_b_id > 0
    assert model_a_id != model_b_id


def test_model_a_metrics_seeded(registered_models):
    model_a_id, _ = registered_models
    resp = requests.get(f"{TRACKER_URL}/models/{model_a_id}/metrics", timeout=5)
    assert resp.status_code == 200
    metrics = {e["metric_name"]: e["metric_value"] for e in resp.json()["evaluations"]}
    assert "accuracy" in metrics
    assert metrics["accuracy"] == pytest.approx(0.912)


def test_model_b_metrics_seeded(registered_models):
    _, model_b_id = registered_models
    resp = requests.get(f"{TRACKER_URL}/models/{model_b_id}/metrics", timeout=5)
    assert resp.status_code == 200
    metrics = {e["metric_name"]: e["metric_value"] for e in resp.json()["evaluations"]}
    assert "accuracy" in metrics
    assert metrics["accuracy"] == pytest.approx(0.934)


# Step 3: A/B configuration


def test_ab_config_is_active(ab_config):
    assert ab_config["active"] is True


def test_ab_config_split_weight(ab_config):
    assert ab_config["split_weight"] == pytest.approx(SPLIT_WEIGHT)


def test_active_ab_endpoint_returns_config(ab_config):
    resp = requests.get(f"{TRACKER_URL}/ab/active", timeout=5)
    assert resp.status_code == 200
    assert resp.json()["active"] is True


# Step 4 & 5: 100 requests — verify traffic split within ±10%


def test_predict_all_requests_succeed(predict_results):
    assert len(predict_results) == TOTAL_REQUESTS


def test_predict_responses_have_model_version(predict_results):
    for result in predict_results:
        assert "model_version" in result
        assert result["model_version"] in ("model_a", "model_b", "primary")


def test_predict_confidence_always_in_range(predict_results):
    for result in predict_results:
        assert 0.0 <= result["confidence"] <= 1.0


def test_traffic_split_within_tolerance(predict_results):
    """
    Core assertion: routing ratio must be within ±10% of split_weight.
    e.g. for split_weight=0.6: model_a should get 50%–70% of traffic.
    """
    model_a_count = sum(1 for r in predict_results if r["model_version"] == "model_a")
    model_b_count = sum(1 for r in predict_results if r["model_version"] == "model_b")
    primary_count = sum(1 for r in predict_results if r["model_version"] == "primary")

    print(f"\n  model_a: {model_a_count}/{TOTAL_REQUESTS} ({model_a_count / TOTAL_REQUESTS:.1%})")
    print(f"  model_b: {model_b_count}/{TOTAL_REQUESTS} ({model_b_count / TOTAL_REQUESTS:.1%})")
    print(f"  primary: {primary_count}/{TOTAL_REQUESTS} (fallback — tracker unreachable)")

    if primary_count == TOTAL_REQUESTS:
        pytest.skip("All requests fell back to primary — is the Experiment Tracker running?")

    routed = model_a_count + model_b_count
    actual_ratio = model_a_count / routed if routed > 0 else 0.0
    lower = SPLIT_WEIGHT - TOLERANCE
    upper = SPLIT_WEIGHT + TOLERANCE

    assert lower <= actual_ratio <= upper, (
        f"Traffic split {actual_ratio:.1%} outside [{lower:.0%}, {upper:.0%}] "
        f"(model_a={model_a_count}, model_b={model_b_count})"
    )


# Step 6: /ab/compare returns correct side-by-side metrics


def test_ab_compare_returns_200(ab_config):
    resp = requests.get(f"{TRACKER_URL}/ab/compare", timeout=5)
    assert resp.status_code == 200


def test_ab_compare_model_b_has_higher_accuracy(ab_config):
    resp = requests.get(f"{TRACKER_URL}/ab/compare", timeout=5)
    body = resp.json()
    assert body["model_b"]["metrics"]["accuracy"] > body["model_a"]["metrics"]["accuracy"]


def test_ab_compare_model_b_has_lower_latency(ab_config):
    resp = requests.get(f"{TRACKER_URL}/ab/compare", timeout=5)
    body = resp.json()
    assert (
        body["model_b"]["metrics"]["latency_p95_ms"] < body["model_a"]["metrics"]["latency_p95_ms"]
    )


def test_ab_compare_split_weight_matches_config(ab_config):
    resp = requests.get(f"{TRACKER_URL}/ab/compare", timeout=5)
    assert resp.json()["split_weight"] == pytest.approx(SPLIT_WEIGHT)
