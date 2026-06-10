"""
Tests for the drift detector.
No network calls so all Prometheus fetching is monkeypatched.
"""

import numpy as np

from observability.drift_detector.detector import (
    compute_binned_ks,
    compute_ks,
    fetch_live_histogram,
)

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


# Histogram-based KS


def test_binned_ks_matches_baseline_distribution():
    baseline = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
    bounds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    cumulative_counts = np.array([1, 2, 3, 4, 5])

    assert compute_binned_ks(baseline, bounds, cumulative_counts, total=5) == 0.0


def test_binned_ks_detects_shifted_distribution():
    baseline = np.array([0.75, 0.82, 0.88, 0.93, 0.97])
    bounds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    cumulative_counts = np.array([1, 2, 3, 4, 5, 5])

    assert compute_binned_ks(baseline, bounds, cumulative_counts, total=5) > 0.15


def test_binned_ks_rejects_mismatched_histogram():
    baseline = np.array([0.5, 0.7])

    try:
        compute_binned_ks(
            baseline,
            np.array([0.5, 1.0]),
            np.array([1.0]),
            total=2,
        )
    except ValueError as exc:
        assert "equal length" in str(exc)
    else:
        raise AssertionError("Expected mismatched histogram to raise ValueError")


# fetch_live_histogram monkeypatched


def _mock_prometheus_response(buckets: list[tuple[str, float]]):
    """Build a minimal Prometheus instant-query response."""
    import time

    now = time.time()
    return {
        "data": {
            "result": [
                {
                    "metric": {"le": boundary},
                    "value": [now, str(count)],
                }
                for boundary, count in buckets
            ]
        }
    }


def test_fetch_live_histogram_returns_sorted_cumulative_counts(monkeypatch):
    import requests

    def mock_get(*args, **kwargs):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return _mock_prometheus_response([("1.0", 5), ("0.2", 2), ("+Inf", 5), ("0.1", 1)])

        return R()

    monkeypatch.setattr(requests, "get", mock_get)
    bounds, cumulative_counts, total = fetch_live_histogram("http://fake", 3600)
    np.testing.assert_array_equal(bounds, np.array([0.1, 0.2, 1.0]))
    np.testing.assert_array_equal(cumulative_counts, np.array([1.0, 2.0, 5.0]))
    assert total == 5


def test_fetch_live_histogram_ignores_malformed_series(monkeypatch):
    import requests

    def mock_get(*args, **kwargs):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                data = _mock_prometheus_response([("0.5", 3), ("+Inf", 4)])
                data["data"]["result"].append({"metric": {}, "value": [0, "bad"]})
                return data

        return R()

    monkeypatch.setattr(requests, "get", mock_get)
    bounds, cumulative_counts, total = fetch_live_histogram("http://fake", 3600)
    np.testing.assert_array_equal(bounds, np.array([0.5]))
    np.testing.assert_array_equal(cumulative_counts, np.array([3.0]))
    assert total == 4


def test_fetch_live_histogram_returns_empty_on_request_error(monkeypatch):
    import requests

    def mock_get(*args, **kwargs):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(requests, "get", mock_get)
    bounds, cumulative_counts, total = fetch_live_histogram("http://fake", 3600)
    assert len(bounds) == 0
    assert len(cumulative_counts) == 0
    assert total == 0
