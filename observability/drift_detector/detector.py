"""
drift_detector/detector.py

Standalone worker that runs every 60 seconds and computes a
Kolmogorov-Smirnov statistic comparing the live prediction confidence
distribution against the baseline recorded at model registration time.

Emits two Prometheus metrics on its own /metrics endpoint:
  - model_drift_ks_statistic   (Gauge) : KS statistic, 0.0–1.0
  - model_drift_window_size    (Gauge) : number of samples in the live window

If KS-statistic exceeds DRIFT_THRESHOLD (default 0.15), logs a warning.
The alert rule in prometheus.yaml fires a Prometheus alert at the same threshold.

Environment variables:
  PROMETHEUS_URL      URL of the Prometheus server (default: http://prometheus.monitoring.svc.cluster.local:9090)
  BASELINE_PATH       Path to a .npy file containing the baseline confidence array
  DRIFT_THRESHOLD     KS-statistic threshold for drift alerts (default: 0.15)
  WINDOW_SECONDS      How far back to pull live samples (default: 3600 = 1 hour)
  POLL_INTERVAL       Seconds between detector runs (default: 60)
  PORT                Port for the /metrics exposition server (default: 9091)
"""

import logging
import os
import time
from threading import Thread

import numpy as np
import requests
from prometheus_client import Gauge, start_http_server
from scipy.stats import ks_2samp

# Configuration

PROMETHEUS_URL = os.environ.get(
    "PROMETHEUS_URL",
    "http://prometheus.monitoring.svc.cluster.local:9090",
)
BASELINE_PATH = os.environ.get("BASELINE_PATH", "/app/baseline.npy")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.15"))
WINDOW_SECONDS = int(os.environ.get("WINDOW_SECONDS", "3600"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "60"))
PORT = int(os.environ.get("PORT", "9091"))

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [drift-detector] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Prometheus metrics emitted by this worker

KS_STATISTIC = Gauge(
    "model_drift_ks_statistic",
    "KS-test statistic comparing live vs baseline confidence distribution",
)

WINDOW_SIZE = Gauge(
    "model_drift_window_size",
    "Number of confidence score samples in the current live window",
)

# Baseline loading


def load_baseline(path: str) -> np.ndarray:
    """
    Load the baseline confidence distribution from disk.
    The baseline is a 1-D numpy array of confidence scores recorded
    at model registration time (saved by register_baseline.py).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Baseline file not found: {path}. Run register_baseline.py after deploying the model."
        )
    baseline = np.load(path)
    log.info("Loaded baseline: %d samples, mean=%.4f", len(baseline), baseline.mean())
    return baseline


# Live histogram fetching from Prometheus


def fetch_live_histogram(
    prometheus_url: str,
    window_seconds: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Query cumulative prediction_confidence histogram increases over the window.

    Returns finite bucket bounds, cumulative counts, and total observations.
    """
    query = f"sum(increase(prediction_confidence_bucket[{window_seconds}s])) by (le)"

    try:
        resp = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        buckets = []
        total = 0
        for series in data.get("data", {}).get("result", []):
            try:
                boundary = series["metric"]["le"]
                count = max(float(series["value"][1]), 0.0)
                if boundary == "+Inf":
                    total = max(total, int(round(count)))
                else:
                    buckets.append((float(boundary), count))
            except (KeyError, IndexError, TypeError, ValueError):
                continue

        if not buckets:
            return np.array([]), np.array([]), total

        buckets.sort(key=lambda item: item[0])
        bounds = np.array([boundary for boundary, _ in buckets])
        cumulative_counts = np.maximum.accumulate(np.array([count for _, count in buckets]))
        if total == 0:
            total = int(round(cumulative_counts[-1]))

        log.info("Fetched live confidence histogram with %d observations", total)
        return bounds, cumulative_counts, total

    except requests.RequestException as exc:
        log.warning("Failed to fetch live histogram from Prometheus: %s", exc)
        return np.array([]), np.array([]), 0


# KS-test


def compute_ks(baseline: np.ndarray, live: np.ndarray) -> tuple[float, float]:
    """
    Run a two-sample KS test.
    Returns (ks_statistic, p_value).
    """
    result = ks_2samp(baseline, live)
    return float(result.statistic), float(result.pvalue)


def compute_binned_ks(
    baseline: np.ndarray,
    bounds: np.ndarray,
    cumulative_counts: np.ndarray,
    total: int,
) -> float:
    """Compute the maximum baseline/live CDF difference at histogram bounds."""
    if len(baseline) == 0 or len(bounds) == 0 or total <= 0:
        raise ValueError("Baseline and live histogram must be non-empty")
    if len(bounds) != len(cumulative_counts):
        raise ValueError("Histogram bounds and counts must have equal length")

    baseline_cdf = np.searchsorted(np.sort(baseline), bounds, side="right") / len(baseline)
    live_cdf = np.clip(cumulative_counts / total, 0.0, 1.0)
    return float(np.max(np.abs(baseline_cdf - live_cdf)))


# Main detector loop


def run_detector(baseline: np.ndarray) -> None:
    """Run the drift detection loop indefinitely."""
    log.info(
        "Drift detector started. threshold=%.2f interval=%ds",
        DRIFT_THRESHOLD,
        POLL_INTERVAL,
    )

    while True:
        try:
            bounds, cumulative_counts, sample_count = fetch_live_histogram(
                PROMETHEUS_URL,
                WINDOW_SECONDS,
            )

            if sample_count < 30:
                log.info(
                    "Not enough live samples (%d < 30) — skipping KS test",
                    sample_count,
                )
                WINDOW_SIZE.set(sample_count)
                time.sleep(POLL_INTERVAL)
                continue

            ks_stat = compute_binned_ks(
                baseline,
                bounds,
                cumulative_counts,
                sample_count,
            )

            KS_STATISTIC.set(ks_stat)
            WINDOW_SIZE.set(sample_count)

            if ks_stat > DRIFT_THRESHOLD:
                log.warning(
                    "DRIFT DETECTED: KS=%.4f (threshold=%.2f, samples=%d)",
                    ks_stat,
                    DRIFT_THRESHOLD,
                    sample_count,
                )
            else:
                log.info(
                    "No drift: KS=%.4f (samples=%d)",
                    ks_stat,
                    sample_count,
                )

        except Exception as exc:  # noqa: BLE001
            log.error("Detector loop error: %s", exc, exc_info=True)

        time.sleep(POLL_INTERVAL)


# Entrypoint


if __name__ == "__main__":
    baseline = load_baseline(BASELINE_PATH)

    # Start Prometheus metrics server in a background thread
    log.info("Starting metrics server on port %d", PORT)
    Thread(target=start_http_server, args=(PORT,), daemon=True).start()

    run_detector(baseline)
