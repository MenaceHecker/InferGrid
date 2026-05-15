from prometheus_client import Counter, Histogram, make_asgi_app

# Metrics

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the inference API",
    # labelnames let Prometheus filter by endpoint and status code
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "End-to-end request latency in seconds",
    ["endpoint"],
    # p50 / p95 / p99 buckets
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Distribution of model prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ASGI sub-app to be mounted at /metrics in main.py

metrics_app = make_asgi_app()