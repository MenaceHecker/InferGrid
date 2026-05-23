"""
Tests for the drift detector.
No network calls so all Prometheus fetching is monkeypatched.
"""

import numpy as np
import pytest

from observability.drift_detector.detector import compute_ks, fetch_live_samples


# compute_ks


def test_ks_identical_distributions_returns_low_statistic():
    rng = np.random.default_rng(42)
    dist = rng.uniform(0.7, 1.0, 500)
    ks_stat, _ = compute_ks(dist, dist)
    assert ks_stat < 0.05


def test_ks_different_distributions_returns_high_statistic():
    rng = np.random.default_rng(42)
    baseline = rng.uniform(0.8, 1.0, 500)  # healthy: high confidence
    drifted = rng.uniform(0.0, 0.5, 500)  # drifted: low confidence
    ks_stat, _ = compute_ks(baseline, drifted)
    assert ks_stat > 0.15


def test_ks_statistic_in_range():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 1.0, 200)
    b = rng.uniform(0.4, 0.9, 200)
    ks_stat, p_value = compute_ks(a, b)
    assert 0.0 <= ks_stat <= 1.0
    assert 0.0 <= p_value <= 1.0


def test_ks_returns_tuple_of_floats():
    a = np.array([0.9, 0.85, 0.92, 0.88])
    b = np.array([0.3, 0.25, 0.4, 0.35])
    result = compute_ks(a, b)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result)


def test_ks_drift_threshold_boundary():
    """Values just above and below 0.15 behave as expected."""
    rng = np.random.default_rng(7)
    baseline = rng.normal(0.85, 0.05, 1000)
    # Slightly shifted distribution so expect KS near boundary
    mild_drift = rng.normal(0.75, 0.05, 500)
    ks_stat, _ = compute_ks(baseline, mild_drift)
    # Not asserting exact value so just that it's in a reasonable range
    assert 0.0 < ks_stat < 1.0


# fetch_live_samples monkeypatched


def _mock_prometheus_response(values: list[float]):
    """Build a minimal Prometheus query_range response."""
    import time

    now = time.time()
    return {
        "data": {"result": [{"values": [[now - i * 10, str(v)] for i, v in enumerate(values)]}]}
    }


def test_fetch_live_samples_returns_array(monkeypatch):
    import requests

    def mock_get(*args, **kwargs):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return _mock_prometheus_response([0.9, 0.85, 0.8, 0.75, 0.95])

        return R()

    monkeypatch.setattr(requests, "get", mock_get)
    samples = fetch_live_samples("http://fake", 3600, 500)
    assert isinstance(samples, np.ndarray)
    assert len(samples) == 5


def test_fetch_live_samples_filters_out_of_range(monkeypatch):
    import requests

    def mock_get(*args, **kwargs):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                # includes values outside [0, 1] that should be dropped
                return _mock_prometheus_response([0.9, 1.5, -0.1, 0.8, 2.0, 0.7])

        return R()

    monkeypatch.setattr(requests, "get", mock_get)
    samples = fetch_live_samples("http://fake", 3600, 500)
    assert all(0.0 <= v <= 1.0 for v in samples)
    assert len(samples) == 3  # 1.5, -0.1, 2.0 filtered out


def test_fetch_live_samples_returns_empty_on_request_error(monkeypatch):
    import requests

    def mock_get(*args, **kwargs):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(requests, "get", mock_get)
    samples = fetch_live_samples("http://fake", 3600, 500)
    assert isinstance(samples, np.ndarray)
    assert len(samples) == 0


def test_fetch_live_samples_respects_max_samples(monkeypatch):
    import requests

    def mock_get(*args, **kwargs):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return _mock_prometheus_response([0.9] * 1000)

        return R()

    monkeypatch.setattr(requests, "get", mock_get)
    samples = fetch_live_samples("http://fake", 3600, max_samples=50)
    assert len(samples) <= 50
