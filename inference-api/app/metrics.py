"""
Prometheus instrumentation for the InferGrid inference API.

The /metrics endpoint is handled by mounting metrics_app in main.py.
Prometheus scrapes it via the pod annotation prometheus.io/scrape: "true".
"""

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the inference API",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "End-to-end request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Distribution of model prediction confidence scores (0.0 to 1.0)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Wall-clock seconds to load the model at startup",
)

AB_ROUTING_COUNT = Counter(
    "ab_routing_total",
    "Total predictions routed per model version in A/B test",
    ["model_version"],
)

metrics_app = make_asgi_app()
