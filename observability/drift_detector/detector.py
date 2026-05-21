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