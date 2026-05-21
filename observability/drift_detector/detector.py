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
  WINDOW_SAMPLES      Max samples to pull from Prometheus (default: 500)
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
WINDOW_SAMPLES = int(os.environ.get("WINDOW_SAMPLES", "500"))
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


# Live sample fetching from Prometheus


def fetch_live_samples(
    prometheus_url: str,
    window_seconds: int,
    max_samples: int,
) -> np.ndarray:
    """
    Query Prometheus for recent prediction_confidence observations.
    Uses the _sum / _count approach to approximate the mean per bucket,
    then falls back to pulling raw bucket boundaries as a distribution.

    Returns a 1-D numpy array of confidence scores (may be empty).
    """
    end = time.time()
    start = end - window_seconds
    step = max(window_seconds // max_samples, 1)

    query = "rate(prediction_confidence_sum[2m]) / rate(prediction_confidence_count[2m])"

    try:
        resp = requests.get(
            f"{prometheus_url}/api/v1/query_range",
            params={
                "query": query,
                "start": start,
                "end": end,
                "step": step,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        values = []
        for series in data.get("data", {}).get("result", []):
            for _, v in series.get("values", []):
                try:
                    fv = float(v)
                    if 0.0 <= fv <= 1.0:
                        values.append(fv)
                except (ValueError, TypeError):
                    continue

        samples = np.array(values[:max_samples])
        log.info("Fetched %d live samples from Prometheus", len(samples))
        return samples

    except requests.RequestException as exc:
        log.warning("Failed to fetch live samples from Prometheus: %s", exc)
        return np.array([])


# KS-test


def compute_ks(baseline: np.ndarray, live: np.ndarray) -> tuple[float, float]:
    """
    Run a two-sample KS test.
    Returns (ks_statistic, p_value).
    """
    result = ks_2samp(baseline, live)
    return float(result.statistic), float(result.pvalue)


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
            live = fetch_live_samples(PROMETHEUS_URL, WINDOW_SECONDS, WINDOW_SAMPLES)

            if len(live) < 30:
                log.info("Not enough live samples (%d < 30) — skipping KS test", len(live))
                WINDOW_SIZE.set(len(live))
                time.sleep(POLL_INTERVAL)
                continue

            ks_stat, p_value = compute_ks(baseline, live)

            KS_STATISTIC.set(ks_stat)
            WINDOW_SIZE.set(len(live))

            if ks_stat > DRIFT_THRESHOLD:
                log.warning(
                    "DRIFT DETECTED: KS=%.4f p=%.4f (threshold=%.2f, samples=%d)",
                    ks_stat,
                    p_value,
                    DRIFT_THRESHOLD,
                    len(live),
                )
            else:
                log.info(
                    "No drift: KS=%.4f p=%.4f (samples=%d)",
                    ks_stat,
                    p_value,
                    len(live),
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
